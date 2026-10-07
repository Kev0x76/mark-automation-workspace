"""图片型 PDF 的泛化视觉单元分割（视觉候选切块）。

设计原则（V5）：
- 泛化优先：不硬编码任何供应商 / 货号 / 品名 / 固定坐标。
- 仅做"视觉候选切块"，不做字段抽取、不做核对结论。
- 所有候选一律标记为 VISUAL_CELL_CANDIDATE，关联状态 NEEDS_MANUAL_REVIEW。
- 优雅降级：缺少 OpenCV / numpy 时返回空列表，绝不抛出。

核心算法是有界递归 XY-cut（投影白条切割）：
1. 二值化后，按行方向找"近空白带"（白条）把页面切成若干水平块；
2. 在每个水平块内，按列方向找白条切成竖块；
3. 行列交替递归，直到块足够小或再也切不动为止。
该方法只依赖"内容之间存在空白间隔"这一通用排版事实，
对规则标签网格泛化良好，且与具体内容无关。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - 环境缺库时整体降级
    cv2 = None  # type: ignore
    np = None  # type: ignore


@dataclass
class VisualCell:
    """一个视觉候选单元。坐标均为渲染图像素坐标。"""

    index: int
    bbox_px: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    ink_ratio: float                    # 单元内墨色像素占比，粗略质量信号
    area_ratio: float                   # 单元面积占整页比例

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "bbox_px": list(self.bbox_px),
            "ink_ratio": round(self.ink_ratio, 4),
            "area_ratio": round(self.area_ratio, 4),
        }


@dataclass
class VisualSegmentation:
    """整页分割结果，附带一个可比较的质量指标，供熔断机制使用。"""

    cells: list[VisualCell] = field(default_factory=list)
    image_w: int = 0
    image_h: int = 0
    reason: str = ""                    # 未产生有效候选时的原因
    quality: float = 0.0                # 越大越好：候选数 × 内容覆盖率

    @property
    def ok(self) -> bool:
        return bool(self.cells)


# ---- 可调参数（全部相对于图像尺寸，保持分辨率无关、泛化）----
_MAX_DEPTH = 6
_ROW_INK_THR = 0.012      # 行白条判定：该行墨色占比低于此值视为空白
_COL_INK_THR = 0.020      # 列白条判定
_ROW_GAP_FRAC = 0.012     # 行白条最小高度（占页高比例）
_COL_GAP_FRAC = 0.012     # 列白条最小宽度（占页宽比例）
_MIN_W_FRAC = 0.06        # 候选单元最小宽度
_MIN_H_FRAC = 0.04        # 候选单元最小高度
_MIN_CELLS = 2            # 少于此数视为"未分出网格"，不优于整页兜底
_MAX_CELLS = 400          # 多于此数视为过度切割，判为不可靠


def segment_image_page(image: Any) -> VisualSegmentation:
    """对一张渲染好的页面图做视觉候选切块。

    参数 image 可以是 numpy 数组（灰度或 BGR/RGB）或 PIL.Image。
    返回 VisualSegmentation；任何异常或缺库都返回带 reason 的空结果。
    """
    if cv2 is None or np is None:
        return VisualSegmentation(reason="OPENCV_UNAVAILABLE")

    gray = _to_gray(image)
    if gray is None:
        return VisualSegmentation(reason="IMAGE_UNREADABLE")

    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return VisualSegmentation(image_w=w, image_h=h, reason="IMAGE_TOO_SMALL")

    # Otsu 二值化，墨色=1
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (bw > 0).astype(np.float32)

    page_ink = float(ink.mean())
    if page_ink < 0.002:
        return VisualSegmentation(image_w=w, image_h=h, reason="PAGE_NEARLY_BLANK")

    min_w = w * _MIN_W_FRAC
    min_h = h * _MIN_H_FRAC

    raw: list[tuple[int, int, int, int]] = []
    _xy_cut(ink, 0, 0, w, h, axis="h", depth=0,
            page_w=w, page_h=h, min_w=min_w, min_h=min_h, out=raw)

    cells: list[VisualCell] = []
    page_area = float(w * h)
    for (x0, y0, x1, y1) in raw:
        if (x1 - x0) < min_w or (y1 - y0) < min_h:
            continue
        sub = ink[y0:y1, x0:x1]
        cell_ink = float(sub.mean()) if sub.size else 0.0
        if cell_ink < 0.004:  # 几乎空白的块丢弃
            continue
        cells.append(
            VisualCell(
                index=0,
                bbox_px=(x0, y0, x1, y1),
                ink_ratio=cell_ink,
                area_ratio=((x1 - x0) * (y1 - y0)) / page_area,
            )
        )

    # 阅读顺序排序：先上后下、先左后右（按行带聚类）
    cells = _sort_reading_order(cells, h)
    for i, c in enumerate(cells):
        c.index = i + 1

    seg = VisualSegmentation(cells=cells, image_w=w, image_h=h)

    if len(cells) < _MIN_CELLS:
        seg.cells = []
        seg.reason = "NO_GRID_DETECTED"
        return seg
    if len(cells) > _MAX_CELLS:
        seg.cells = []
        seg.reason = "OVER_SEGMENTED"
        return seg

    coverage = sum(c.area_ratio for c in cells)
    seg.quality = len(cells) * min(1.0, coverage)
    return seg


def _to_gray(image: Any):
    try:
        arr = np.asarray(image)
    except Exception:
        return None
    if arr is None or getattr(arr, "size", 0) == 0:
        return None
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        # PIL 转 numpy 为 RGB；cv2 读取为 BGR。灰度转换对二值化无差异。
        channels = arr.shape[2]
        if channels >= 3:
            return cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return arr[:, :, 0].astype(np.uint8)
    return None


def _bands(profile, ink_thr: float, min_gap: int) -> list[list[int]]:
    """在一维墨色占比序列上，返回内容带 [start, end)，
    相邻内容带之间若空白小于 min_gap 则合并。"""
    low = profile < ink_thr
    n = len(low)
    runs: list[list[int]] = []
    i = 0
    while i < n:
        if not low[i]:
            j = i
            while j < n and not low[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def _xy_cut(ink, x0, y0, x1, y1, axis, depth, page_w, page_h, min_w, min_h, out):
    w = x1 - x0
    h = y1 - y0
    if depth > _MAX_DEPTH or w < min_w or h < min_h:
        out.append((x0, y0, x1, y1))
        return

    sub = ink[y0:y1, x0:x1]

    if axis == "h":
        prof = sub.sum(axis=1) / max(1, w)
        bs = [b for b in _bands(prof, _ROW_INK_THR, max(1, int(page_h * _ROW_GAP_FRAC)))
              if (b[1] - b[0]) > min_h * 0.6]
        if len(bs) <= 1:
            _xy_cut(ink, x0, y0, x1, y1, "v", depth + 1,
                    page_w, page_h, min_w, min_h, out)
            return
        for s, e in bs:
            _xy_cut(ink, x0, y0 + s, x1, y0 + e, "v", depth + 1,
                    page_w, page_h, min_w, min_h, out)
    else:
        prof = sub.sum(axis=0) / max(1, h)
        bs = [b for b in _bands(prof, _COL_INK_THR, max(1, int(page_w * _COL_GAP_FRAC)))
              if (b[1] - b[0]) > min_w * 0.6]
        if len(bs) <= 1:
            # 列切不动，再尝试一次行切；仍切不动则作为最终块输出。
            prof2 = sub.sum(axis=1) / max(1, w)
            bs2 = [b for b in _bands(prof2, _ROW_INK_THR, max(1, int(page_h * _ROW_GAP_FRAC)))
                   if (b[1] - b[0]) > min_h * 0.6]
            if len(bs2) <= 1:
                out.append((x0, y0, x1, y1))
                return
            for s, e in bs2:
                _xy_cut(ink, x0, y0 + s, x1, y0 + e, "v", depth + 1,
                        page_w, page_h, min_w, min_h, out)
            return
        for s, e in bs:
            _xy_cut(ink, x0 + s, y0, x0 + e, y1, "h", depth + 1,
                    page_w, page_h, min_w, min_h, out)


def _sort_reading_order(cells: list[VisualCell], page_h: int) -> list[VisualCell]:
    if not cells:
        return cells
    # 用单元高度的一半作为行聚类容差，把同一行的单元归到一起再按 x 排序
    median_h = sorted((c.bbox_px[3] - c.bbox_px[1]) for c in cells)[len(cells) // 2]
    tol = max(10, median_h * 0.5)
    ordered = sorted(cells, key=lambda c: c.bbox_px[1])
    rows: list[list[VisualCell]] = []
    for c in ordered:
        top = c.bbox_px[1]
        if rows and abs(rows[-1][0].bbox_px[1] - top) <= tol:
            rows[-1].append(c)
        else:
            rows.append([c])
    result: list[VisualCell] = []
    for row in rows:
        row.sort(key=lambda c: c.bbox_px[0])
        result.extend(row)
    return result


# ---------------------------------------------------------------------------
# 候选生成 + manifest 写入（增量、纯增加产物，不触碰既有 manifest / 核对结果）
# ---------------------------------------------------------------------------

#: 唯一允许的候选状态，明确区别于 OK / PASS，避免被误判为核对通过。
VISUAL_CELL_STATUS = "VISUAL_CELL_CANDIDATE"
VISUAL_CELL_ASSOCIATION = "NEEDS_MANUAL_REVIEW"
_VISUAL_CELL_MESSAGE = (
    "视觉候选切块：仅按版面空白自动切出的候选区域，未做字段抽取、未做关联、"
    "未做核对。请人工复核，不能视为核对通过。"
)


def generate_visual_candidates(output_dir, pages, breaker=None) -> dict:
    """对图片型（无文本层）页面生成视觉候选切块及 manifest。

    这是一个**纯增量**步骤：只新增 ``visual_cells/`` 目录与
    ``visual_cells_manifest.json``，不修改任何既有产物或核对结论。
    任何异常都被吞掉并记录到 manifest 的 events，绝不影响主流程。

    返回写入的 manifest dict。
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    asset_dir = output_dir / "visual_cells"
    items: list[dict[str, Any]] = []
    events: list[str] = []
    pages_processed = 0

    try:
        from PIL import Image  # 延迟导入，缺库则降级
    except Exception:
        Image = None  # type: ignore

    def _do() -> None:
        nonlocal pages_processed
        if Image is None or cv2 is None or np is None:
            events.append("VISUAL_SEGMENTATION_SKIPPED:依赖缺失(PIL/OpenCV/numpy)")
            return
        for page in pages:
            # 只处理图片型页面：无文本层。文本型走既有 text grid 路径，互不干扰。
            if getattr(page, "has_text_layer", True):
                continue
            img = getattr(page, "render_image", None)
            if img is None:
                events.append(
                    f"VISUAL_SEGMENTATION_NO_RENDER:{Path(getattr(page,'file_path','')).name}"
                    f":p{getattr(page,'page',0) + 1}"
                )
                continue
            seg = segment_image_page(img)
            pages_processed += 1
            if not seg.ok:
                events.append(
                    f"VISUAL_SEGMENTATION_NO_CANDIDATES:{Path(getattr(page,'file_path','')).name}"
                    f":p{getattr(page,'page',0) + 1}:{seg.reason or 'UNKNOWN'}"
                )
                continue
            asset_dir.mkdir(parents=True, exist_ok=True)
            src_name = Path(getattr(page, "file_path", "mark")).stem
            page_no = getattr(page, "page", 0) + 1
            crop_source = img
            scale_x = 1.0
            scale_y = 1.0
            high_res = _render_pdf_page_image(getattr(page, "file_path", ""), getattr(page, "page", 0))
            if high_res is not None:
                crop_source = high_res
                scale_x = high_res.width / max(seg.image_w, 1)
                scale_y = high_res.height / max(seg.image_h, 1)
            for cell in seg.cells:
                x0, y0, x1, y1 = cell.bbox_px
                crop_name = f"{_safe(src_name)}_p{page_no}_cell_{cell.index:03d}.png"
                crop_path = asset_dir / crop_name
                try:
                    crop_box = (
                        round(x0 * scale_x),
                        round(y0 * scale_y),
                        round(x1 * scale_x),
                        round(y1 * scale_y),
                    )
                    crop_source.crop(crop_box).save(crop_path)
                except Exception as exc:
                    events.append(f"VISUAL_CELL_CROP_FAILED:{crop_name}:{type(exc).__name__}")
                    continue
                items.append({
                    "source": str(getattr(page, "file_path", "")),
                    "page": page_no,
                    "cell_index": cell.index,
                    "status": VISUAL_CELL_STATUS,
                    "association": VISUAL_CELL_ASSOCIATION,
                    "image": str(crop_path),
                    "bbox_px": list(cell.bbox_px),
                    "image_size_px": [seg.image_w, seg.image_h],
                    "ink_ratio": round(cell.ink_ratio, 4),
                    "area_ratio": round(cell.area_ratio, 4),
                    "quality": round(seg.quality, 3),
                    "message": _VISUAL_CELL_MESSAGE,
                })

    if breaker is not None and hasattr(breaker, "run"):
        breaker.run("visual_segmentation", _do, None)
        events.extend(getattr(breaker, "events", []))
    else:
        try:
            _do()
        except Exception as exc:  # 最后一道防线
            events.append(f"VISUAL_SEGMENTATION_ERROR:{type(exc).__name__}:{exc}")

    manifest = {
        "asset_dir": str(asset_dir),
        "note": _VISUAL_CELL_MESSAGE,
        "pages_processed": pages_processed,
        "candidate_count": len(items),
        "items": items,
        "events": events,
    }
    try:
        import json
        (output_dir / "visual_cells_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        manifest.setdefault("events", []).append(
            f"VISUAL_MANIFEST_WRITE_FAILED:{type(exc).__name__}:{exc}"
        )
    return manifest


def _safe(value: str) -> str:
    keep = [ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(value)]
    return "".join(keep).strip("_") or "mark"


def _render_pdf_page_image(file_path: str, page_index: int, zoom: float = 6.0) -> Any:
    """Render a sharper page image for saved candidate crops when possible."""
    from pathlib import Path

    path = Path(str(file_path))
    if path.suffix.lower() != ".pdf" or not path.exists():
        return None
    try:
        import fitz
        from PIL import Image

        doc = fitz.open(path)
        try:
            page = doc[int(page_index)]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()
    except Exception:
        return None
