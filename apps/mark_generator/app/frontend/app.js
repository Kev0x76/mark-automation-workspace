const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const GOLDEN_PROJECT_ID = "golden-candidate-a5-huahong-photo-frame-20260420-v1";
const state = {
  workspace: "contract",
  projectId: "",
  project: null,
  snapshot: null,
  extraction: null,
  manifest: null,
  verificationReport: null,
  verifyProgress: null,
  linearts: { items: [], summary: {} },
  regions: { items: [], summary: {} },
  groups: { suggestions: { groups: [], manual_groups: [] }, approvals: { approvals: [] } },
  progress: null,
  lineartProgress: null,
  providerStatus: null,
  pendingShares: [],
  items: [],
  selectedItemNo: null,
  selectedVerificationItemNo: null,
  selectedGroupId: null,
  grouping: "category",
  expandedGroups: new Set(),
  itemSearch: "",
  taskFilter: "open",
  externalRun: null,
  busy: false,
  intake: null,
  progressTimer: null,
  lineartProgressTimer: null,
  projectPrompt: { prompt_instruction: "", updated_at: null },
  projectPromptDirty: false,
  promptDrafts: {},
};

const manualCropState = {
  image: null,
  rect: null,
  dragStart: null,
  dragging: false,
};

const fieldLabels = {
  item_number: "货号",
  barcode: "条码",
  barcode_scan: "条码扫码",
  barcode_text: "条码明文",
  barcode_scan_vs_text: "扫码与明文",
  packing_qty: "装箱数",
  name_cn: "中文品名",
  name_it: "意大利语品名",
  name_es: "西班牙语品名",
  name_en: "英语品名",
  line_art: "线稿",
  __coverage__: "核对覆盖范围",
  __evidence_archive__: "核对证据归档",
  __weak_evidence__: "弱证据",
  __verification__: "核对总结果",
  __item__: "货号综合结论",
};

const issueStatusLabels = {
  COVERAGE_INCOMPLETE: "覆盖不完整",
  EVIDENCE_ARCHIVE_INCOMPLETE: "证据归档不完整",
  WEAK_EVIDENCE: "存在弱证据",
  VERIFICATION_FAILED: "机器核对未通过",
  BLOCKING_ISSUE: "阻断问题",
  NEEDS_MANUAL_REVIEW: "需要人工审核",
  NEEDS_REVIEW: "需要人工审核",
  MISMATCH: "不一致",
  MISSING: "缺失",
  ERROR: "核对错误",
};

function create(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = value == null || value === "" ? "—" : String(value);
}

function setHidden(selector, hidden) {
  const node = $(selector);
  if (node) node.classList.toggle("hidden", Boolean(hidden));
}

function setDisabled(selector, disabled) {
  const node = $(selector);
  if (node) node.disabled = Boolean(disabled);
}

function formData(fields = {}) {
  const data = new FormData();
  Object.entries(fields).forEach(([key, value]) => {
    if (value !== undefined && value !== null) data.append(key, String(value));
  });
  return data;
}

async function request(url, options = {}) {
  const { timeoutMs = 0, ...fetchOptions } = options;
  let timer = null;
  if (timeoutMs) {
    // A stalled poll on a lossy link must not freeze progress forever.
    const controller = new AbortController();
    fetchOptions.signal = controller.signal;
    timer = window.setTimeout(() => controller.abort(), timeoutMs);
  }
  let response;
  try {
    response = await fetch(url, fetchOptions);
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("网络响应超时，请稍后重试");
    throw new Error("网络连接中断，请检查网络后重试");
  } finally {
    if (timer) window.clearTimeout(timer);
  }
  const raw = await response.text();
  let payload = {};
  try { payload = raw ? JSON.parse(raw) : {}; } catch { payload = raw; }
  if (!response.ok) {
    const detail = typeof payload === "object" && payload ? payload.detail || payload.error : payload;
    const message = typeof detail === "object" && detail
      ? detail.message || detail.code || `请求失败（HTTP ${response.status}）`
      : detail || `请求失败（HTTP ${response.status}）`;
    const error = new Error(message);
    if (typeof detail === "object" && detail) {
      error.code = detail.code || null;
      error.details = detail.details || null;
    }
    throw error;
  }
  return payload;
}

async function optionalRequest(url, fallback = null, options = {}) {
  try { return await request(url, options); } catch { return fallback; }
}

const POLL_TIMEOUT_MS = 20000;

function fileUrl(projectId, relativePath) {
  if (!projectId || !relativePath) return "";
  const path = String(relativePath).replaceAll("\\", "/").split("/").map(encodeURIComponent).join("/");
  return `/api/projects/${encodeURIComponent(projectId)}/files/${path}`;
}

/**
 * Display-size WebP of a project image (server-side downscale, ETag-revalidated).
 * Lightbox and manual crop keep using fileUrl() so inspection sees the original.
 */
function thumbUrl(projectId, relativePath, width) {
  const original = fileUrl(projectId, relativePath);
  return original ? `${original.replace("/files/", "/thumbs/")}?w=${width}` : "";
}

/**
 * One front/side pair. `format: "png"` is the full-fidelity original (lightbox);
 * "webp" without width is a lossless render with identical pixels; with width a light thumbnail.
 * The stamp must stay stable across renders: a changing URL defeats every cache.
 */
function markPreviewUrl(itemNo, { format = "png", width = null } = {}) {
  if (!state.projectId || !itemNo) return "";
  const stamp = state.manifest?.updated_at || state.manifest?.created_at || "none";
  const query = new URLSearchParams({ v: stamp });
  if (width) query.set("w", String(width));
  return `/api/projects/${encodeURIComponent(state.projectId)}/standard-marks/${encodeURIComponent(itemNo)}/preview.${format}?${query}`;
}

function sameUrl(image, url) {
  return Boolean(url) && image.getAttribute("src") === url;
}

/**
 * Point an <img> at `url` with a loading state on its frame; no-op when unchanged
 * so re-renders never restart a download or flash the image.
 * `placeholder` (an already-cached smaller variant) shows instantly while `url` loads.
 */
function setImageSource(image, url, { placeholder = null, onError = null } = {}) {
  const frame = image.parentElement;
  if (sameUrl(image, url) || image.dataset.pendingSrc === url) return;
  // Hidden workspaces re-render on every selection; fetch their images only once shown.
  const view = image.closest(".workspace-view");
  if (view && !view.classList.contains("active")) {
    deferredImageLoads.set(image, () => setImageSource(image, url, { placeholder, onError }));
    return;
  }
  deferredImageLoads.delete(image);
  image.onerror = null;
  image.dataset.pendingSrc = url;
  const settle = () => {
    if (image.dataset.pendingSrc !== url) return;
    delete image.dataset.pendingSrc;
    frame?.classList.remove("is-loading");
  };
  frame?.classList.add("is-loading");
  if (placeholder && placeholder !== url) {
    image.src = placeholder;
    const full = new Image();
    full.decoding = "async";
    full.onload = () => {
      if (image.dataset.pendingSrc !== url) return;
      image.src = url;
      settle();
    };
    full.onerror = () => {
      if (image.dataset.pendingSrc !== url) return;
      settle();
      onError?.();
    };
    full.src = url;
    return;
  }
  image.onload = settle;
  image.onerror = () => {
    if (image.dataset.pendingSrc !== url) return;
    settle();
    onError?.();
  };
  image.src = url;
}

const deferredImageLoads = new Map();

function flushDeferredImages() {
  [...deferredImageLoads].forEach(([image, load]) => {
    if (image.closest(".workspace-view")?.classList.contains("active")) load();
  });
}

function clearImageSource(image) {
  deferredImageLoads.delete(image);
  delete image.dataset.pendingSrc;
  image.onerror = null;
  image.removeAttribute("src");
  image.parentElement?.classList.remove("is-loading");
}

function contractFilename() {
  return state.snapshot?.source_contract?.original_filename
    || state.extraction?.original_filename
    || state.snapshot?.source_contract?.path?.split(/[\\/]/).pop()
    || state.projectId
    || "未命名合同";
}

function contractStem() {
  const original = state.snapshot?.source_contract?.original_filename || state.extraction?.original_filename;
  return original ? String(original).replace(/\.(xlsx|xls)$/i, "") : (state.projectId || "未命名项目");
}

function deliveryFilename() {
  return `${contractStem()}-唛头.pdf`;
}

function contractItemRegionLabel() {
  const region = state.extraction?.contract_item_region;
  const spans = region?.spans || [];
  if (!spans.length) return "未记录";
  const labels = spans.slice(0, 3).map((span) => (
    span.start_row === span.end_row
      ? `第 ${span.start_row} 行`
      : `第 ${span.start_row}–${span.end_row} 行`
  ));
  const ignored = Number(region.ignored_non_product_rows_count || 0);
  return `${labels.join("、")}${spans.length > 3 ? ` 等 ${spans.length} 个区段` : ""}${ignored ? ` · 已排除 ${ignored} 行非商品内容` : ""}`;
}

function statusClass(status) {
  if (status === "manual_release") return "manual";
  if (["ready", "approved", "selected", "matched", "verified_pass", "completed", "OK", "PASS"].includes(status)) return "ready";
  if (["running", "queued", "processing"].includes(status)) return "processing";
  if (["failed", "verified_failed", "rejected", "BLOCKING_ISSUE"].includes(status)) return "failed";
  if (["needs_review", "pending", "stale", "interrupted", "NEEDS_MANUAL_REVIEW"].includes(status)) return "attention";
  return "neutral";
}

function setStatus(selector, text, status = "neutral") {
  const node = $(selector);
  if (!node) return;
  node.textContent = text;
  node.className = `status-badge ${statusClass(status)}`;
}

function notify(title, copy, type = "success") {
  const toast = $("#toast");
  setText("#toast-title", title);
  setText("#toast-copy", copy);
  toast.className = `toast${type === "error" ? " error" : ""}`;
  window.clearTimeout(notify.timer);
  notify.timer = window.setTimeout(() => toast.classList.add("hidden"), 4600);
}

function showNotice(title, copy, type = "warning") {
  setText("#notice-title", title);
  setText("#notice-copy", copy);
  const bar = $("#notice-bar");
  bar.className = `notice-bar${type === "error" ? " error" : type === "success" ? " success" : ""}`;
}

function closeNotice() {
  $("#notice-bar").classList.add("hidden");
}

function setBusy(value) {
  state.busy = Boolean(value);
  document.body.classList.toggle("is-busy", state.busy);
  renderActions();
}

async function runAction(title, action, successCopy = "操作已完成") {
  if (state.busy) return null;
  setBusy(true);
  try {
    const result = await action();
    notify(title, successCopy);
    return result;
  } catch (error) {
    showNotice(title, error.message, "error");
    notify(title, error.message, "error");
    throw error;
  } finally {
    setBusy(false);
  }
}

function openModal(id) {
  const node = document.getElementById(id);
  if (!node) return;
  node.classList.remove("hidden");
  node.querySelector("input, select, textarea, button")?.focus();
}

function closeModal(id) {
  document.getElementById(id)?.classList.add("hidden");
}

function closeAllModals() {
  $$(".modal-backdrop, .image-lightbox").forEach((node) => node.classList.add("hidden"));
}

function showWorkspace(name) {
  state.workspace = name;
  $$(".workspace-tab").forEach((button) => button.classList.toggle("active", button.dataset.workspace === name));
  $$(".workspace-view").forEach((view) => view.classList.toggle("active", view.dataset.view === name));
  flushDeferredImages();
}

function getItem(itemNo = state.selectedItemNo) {
  return state.items.find((item) => String(item.item_no) === String(itemNo)) || null;
}

function allGroups() {
  const suggestions = state.groups?.suggestions || {};
  return [
    ...(suggestions.manual_groups || []).map((group) => ({ ...group, manual: true })),
    ...(suggestions.groups || []).map((group) => ({ ...group, manual: false })),
  ];
}

function selectedGroup() {
  const group = allGroups().find((value) => value.group_id === state.selectedGroupId) || null;
  return group && (group.manual || (group.members?.length || 0) >= 2) ? group : null;
}

function mergeItems(normalizedItems, regionItems, lineartItems) {
  const order = [];
  const byNo = new Map();
  [normalizedItems, regionItems, lineartItems].forEach((source) => {
    (source || []).forEach((item) => {
      const itemNo = String(item.item_no || "").trim();
      if (!itemNo) return;
      if (!byNo.has(itemNo)) order.push(itemNo);
      byNo.set(itemNo, { ...(byNo.get(itemNo) || {}), ...item });
    });
  });
  const regionByNo = new Map((regionItems || []).map((item) => [String(item.item_no), item]));
  return order.map((itemNo) => ({ ...byNo.get(itemNo), regionEntry: regionByNo.get(itemNo) || null }));
}

function itemState(item) {
  if (!item?.source_image || item.source_image_status === "missing") return { label: "缺产品图", state: "attention" };
  if (item.product_region_status !== "selected") return { label: "待选产品区域", state: "processing" };
  if (item.lineart_status === "stale") return { label: "线稿已失效", state: "failed" };
  if (item.lineart_status !== "approved") return { label: "待生成线稿", state: "attention" };
  return { label: "可入版", state: "ready" };
}

function isLineartReady(item) {
  return item?.lineart_status === "approved" && ["passed", "accepted", "human_passed"].includes(item?.lineart_qc_status);
}

/** SKUs with region selected but no approved lineart ready for production. */
function missingLineartItems() {
  return state.items.filter((entry) => entry.product_region_status === "selected" && !isLineartReady(entry));
}

function usesRepStrategy() {
  return Boolean($("#lineart-strategy-reps")?.checked);
}

/**
 * Share targets of categories whose representative already has approved lineart.
 * Under group_representatives, batch will NOT generate these — share is optional only.
 */
function pendingShareTargetNos() {
  const targets = new Set();
  for (const row of state.pendingShares || state.lineartProgress?.pending_shares || []) {
    if (!isLineartReady(getItem(row.representative_item_no))) continue;
    for (const itemNo of row.share_target_item_nos || []) {
      if (itemNo && !isLineartReady(getItem(itemNo))) targets.add(String(itemNo));
    }
  }
  return targets;
}

/** True when every remaining missing SKU is only waiting on optional share, not on rep generation. */
function onlyOptionalShareRemain() {
  const missing = missingLineartItems();
  if (!missing.length) return false;
  const targets = pendingShareTargetNos();
  if (!targets.size) return false;
  return missing.every((entry) => targets.has(String(entry.item_no)));
}

function currentVerificationResult(itemNo = state.selectedItemNo) {
  return (state.verificationReport?.results || []).find((result) => String(result.item_number) === String(itemNo)) || null;
}

function projectVerification() {
  return state.manifest?.verification || null;
}

async function loadProject(projectId, { quiet = false } = {}) {
  const normalizedId = String(projectId || "").trim();
  if (!normalizedId) return;
  if (!quiet) setBusy(true);
  try {
    const base = `/api/projects/${encodeURIComponent(normalizedId)}`;
    // One parallel wave: each extra sequential round trip costs ~0.4 s on the public link.
    const [project, linearts, regions, groups, normalized, snapshot, extraction, manifest, verificationReport, progress, lineartProgress, projectPrompt, verifyProgress] = await Promise.all([
      request(base),
      request(`${base}/linearts`),
      optionalRequest(`${base}/product-regions`, { items: [], summary: {} }),
      optionalRequest(`${base}/lineart-groups`, { suggestions: { groups: [], manual_groups: [] }, approvals: { approvals: [] } }),
      optionalRequest(fileUrl(normalizedId, "data/items.normalized.json"), { items: [] }),
      optionalRequest(fileUrl(normalizedId, "data/contract.snapshot.json"), null),
      optionalRequest(fileUrl(normalizedId, "reports/extraction_report.json"), null),
      optionalRequest(fileUrl(normalizedId, "manifest.json"), null),
      optionalRequest(fileUrl(normalizedId, "reports/verification_report.json"), null),
      optionalRequest(`${base}/product-regions/progress`, null),
      optionalRequest(`${base}/linearts/generate-progress`, null),
      optionalRequest(`${base}/lineart-prompt`, { prompt_instruction: "", updated_at: null }),
      optionalRequest(`${base}/verify/progress`, null),
    ]);
    // Keep the last known job state only when re-reading the same project.
    state.verifyProgress = verifyProgress || (state.projectId === normalizedId ? state.verifyProgress : null);
    state.projectId = normalizedId;
    state.project = project;
    state.linearts = linearts || { items: [], summary: {} };
    state.regions = regions || { items: [], summary: {} };
    state.groups = groups || { suggestions: { groups: [], manual_groups: [] }, approvals: { approvals: [] } };
    state.snapshot = snapshot;
    state.extraction = extraction;
    state.manifest = manifest;
    state.verificationReport = verificationReport;
    state.progress = progress;
    state.lineartProgress = lineartProgress;
    state.projectPrompt = projectPrompt || { prompt_instruction: "", updated_at: null };
    state.projectPromptDirty = false;
    state.items = mergeItems(normalized?.items || [], state.regions.items || [], state.linearts.items || []);
    if (!getItem(state.selectedItemNo)) state.selectedItemNo = state.items[0]?.item_no || null;
    if (!selectedGroup()) {
      const containing = allGroups().find((group) => group.members?.some((member) => String(member.item_no) === String(state.selectedItemNo)));
      state.selectedGroupId = containing?.group_id || allGroups()[0]?.group_id || null;
    }
    if (state.selectedGroupId) state.expandedGroups.add(state.selectedGroupId);
    window.localStorage.setItem("mark-canvas:last-project", normalizedId);
    state.intake = null;
    renderAll();
    if (["queued", "running"].includes(progress?.status)) beginProgressPolling();
    if (["queued", "running"].includes(lineartProgress?.status)) beginLineartProgressPolling();
    if (["queued", "running"].includes(state.verifyProgress?.status)) beginVerifyProgressPolling();
    if (!quiet) notify("项目已打开", `${contractStem()} · ${state.items.length} 个货号`);
  } catch (error) {
    state.projectId = "";
    state.project = null;
    state.items = [];
    renderAll();
    showNotice("项目打开失败", error.message, "error");
    if (!quiet) notify("项目打开失败", error.message, "error");
  } finally {
    if (!quiet) setBusy(false);
  }
}

function renderAll() {
  renderHeader();
  renderContractRail();
  renderGroupTree();
  renderContractWorkspace();
  renderProgress();
  renderLineartWorkspace();
  renderMarkWorkspace();
  renderVerificationWorkspace();
  renderTasks();
  renderActions();
  showWorkspace(state.workspace);
}

function renderSelection() {
  // Selecting a SKU/group must not rebuild contract tables or the whole task rail.
  renderHeader();
  renderGroupTree();
  renderLineartWorkspace();
  renderMarkWorkspace();
  renderVerificationWorkspace();
  renderActions();
}

function renderHeader() {
  const hasProject = Boolean(state.projectId);
  setText("#project-display-name", hasProject ? contractStem() : "尚未打开项目");
  setText("#project-subtitle", hasProject
    ? `项目 ID：${state.projectId} · 对外交付：${deliveryFilename()}`
    : "上传合同后，项目名称与最终 PDF 使用同一合同名称基准。");
  setText("#tab-contract-note", hasProject ? `${state.items.length} 个货号` : "上传与解析");
  const selectedRegions = state.items.filter((item) => item.product_region_status === "selected").length;
  const approved = state.items.filter(isLineartReady).length;
  setText("#tab-lineart-note", hasProject ? `${approved}/${state.items.length} 线稿就绪` : "产品区域与共享");
  setText("#tab-mark-note", state.manifest?.status ? (state.manifest.status === "verified_pass" ? "正式产物已核对" : "正式产物状态已更新") : "正式产物");
  setText("#tab-verify-note", state.project?.readiness?.release_mode === "manual_override" ? "人工确认放行" : state.project?.readiness?.production_ready ? "可交付" : "独立证据门禁");
  $("#contract-state-dot").className = `status-dot ${hasProject ? (selectedRegions === state.items.length ? "ready" : "attention") : ""}`;
}

function renderContractRail() {
  if (!state.projectId) {
    setText("#contract-file-name", "等待上传合同");
    setText("#contract-file-meta", "支持 .xls / .xlsx");
    ["#contract-sheet", "#contract-item-count", "#contract-image-count", "#contract-delivery-name"].forEach((id) => setText(id, "—"));
    return;
  }
  setText("#contract-file-name", contractFilename());
  setText("#contract-file-meta", state.snapshot?.source_contract?.format?.toUpperCase() || state.extraction?.contract_format?.toUpperCase() || "CONTRACT");
  setText("#contract-sheet", state.extraction?.selected_sheet || "未记录");
  setText("#contract-item-count", state.items.length);
  setText("#contract-image-count", `${state.extraction?.matched_images ?? state.items.filter((item) => item.source_image).length}/${state.items.length}`);
  setText("#contract-delivery-name", deliveryFilename());
}

function categoryGroupsForRender() {
  const query = state.itemSearch.toLowerCase();
  const matchesItem = (item) => `${item.item_no || ""} ${item.cn_name || ""}`.toLowerCase().includes(query);
  const useful = allGroups().filter((group) => {
    if (!group.manual && (group.members?.length || 0) < 2) return false;
    if (!query) return true;
    const haystack = [group.normalized_cn_name, ...(group.members || []).flatMap((member) => [member.item_no, member.cn_name])].join(" ").toLowerCase();
    return haystack.includes(query);
  });
  const covered = new Set(useful.flatMap((group) => (group.members || []).map((member) => String(member.item_no))));
  const directItems = state.items.filter((item) => matchesItem(item) && !covered.has(String(item.item_no)));
  if (!useful.length) {
    return directItems.length ? [{
      group_id: "direct-all-items",
      normalized_cn_name: "全部货号",
      members: directItems,
      direct: true,
    }] : [];
  }
  if (directItems.length) {
    useful.push({
      group_id: "direct-ungrouped-items",
      normalized_cn_name: "其他未归类货号",
      members: directItems,
      direct: true,
    });
  }
  return useful;
}

function sequenceGroupsForRender() {
  const query = state.itemSearch.toLowerCase();
  const items = state.items.filter((item) => `${item.item_no} ${item.cn_name || ""}`.toLowerCase().includes(query));
  const groups = [];
  for (let index = 0; index < items.length; index += 50) {
    const members = items.slice(index, index + 50);
    groups.push({
      group_id: `sequence-${index / 50 + 1}`,
      normalized_cn_name: `序号 ${String(index + 1).padStart(3, "0")}–${String(index + members.length).padStart(3, "0")}`,
      members,
      sequence: true,
    });
  }
  return groups;
}

function renderGroupTree() {
  const tree = $("#group-tree");
  tree.replaceChildren();
  const groups = state.grouping === "category" ? categoryGroupsForRender() : sequenceGroupsForRender();
  const visible = new Set(groups.flatMap((group) => (group.members || []).map((member) => String(member.item_no)))).size;
  setText("#visible-item-count", visible);
  $("#group-by-category").classList.toggle("active", state.grouping === "category");
  $("#group-by-sequence").classList.toggle("active", state.grouping === "sequence");
  if (!groups.length) {
    tree.append(create("div", "empty-inline", state.projectId ? "没有匹配的类别或货号。" : "打开项目后显示类别和货号。"));
    return;
  }
  groups.forEach((group) => {
    const wrapper = create("div", "tree-group");
    const head = create("button", "tree-group-button");
    head.type = "button";
    const copy = create("div");
    copy.append(create("strong", "", group.normalized_cn_name || "未命名类别"));
    copy.append(create("small", "", group.direct ? "直接浏览 · 无需先选类别" : group.manual ? "人工类别 · 不自动共享" : group.sequence ? "每批最多 50 个货号" : "按中文品名精确建议"));
    head.append(copy, create("span", "", String(group.members?.length || 0)));
    const expanded = group.direct || state.expandedGroups.has(group.group_id) || state.selectedGroupId === group.group_id;
    const children = create("div", `tree-children${expanded ? "" : " hidden"}`);
    head.addEventListener("click", () => {
      state.selectedGroupId = group.group_id;
      if (state.expandedGroups.has(group.group_id)) state.expandedGroups.delete(group.group_id);
      else state.expandedGroups.add(group.group_id);
      if (!getItem(state.selectedItemNo) || !group.members?.some((member) => String(member.item_no) === String(state.selectedItemNo))) {
        state.selectedItemNo = group.members?.[0]?.item_no || null;
      }
      renderSelection();
    });
    (group.members || []).forEach((member) => {
      const item = getItem(member.item_no) || member;
      const info = itemState(item);
      const button = create("button", `tree-item-button${String(item.item_no) === String(state.selectedItemNo) ? " selected" : ""}`);
      button.type = "button";
      const itemCopy = create("div");
      itemCopy.append(create("strong", "", item.item_no || "未编号"), create("small", "", item.cn_name || "未填写中文品名"));
      button.append(itemCopy, create("span", info.state));
      button.title = info.label;
      button.addEventListener("click", () => {
        state.selectedItemNo = item.item_no;
        state.selectedGroupId = group.group_id;
        renderAll();
      });
      children.append(button);
    });
    wrapper.append(head, children);
    tree.append(wrapper);
  });
}

function renderContractWorkspace() {
  const hasProject = Boolean(state.projectId);
  setText("#summary-total-items", hasProject ? state.items.length : "—");
  setText("#summary-matched-images", hasProject ? `${state.extraction?.matched_images ?? state.items.filter((item) => item.source_image).length}/${state.items.length}` : "—");
  setText("#summary-header-row", state.extraction?.header_row ? `第 ${state.extraction.header_row} 行` : "—");
  setText("#summary-header-note", state.extraction?.columns ? `${Object.keys(state.extraction.columns).length} 个字段 · 合同商品区段 ${contractItemRegionLabel()}` : "失败时停止猜测");
  const nextCode = state.project?.readiness?.next_action?.code;
  const nextActionLabels = {
    complete: "核对通过，可交付 PDF",
    process_product_regions: "继续处理产品区域",
    generate_linearts: "继续生成或补充线稿",
    generate_pdf: "生成正式唛头 PDF",
    verify: "执行独立核对",
  };
  const nextAction = nextActionLabels[nextCode] || (hasProject ? "按任务台账继续" : "上传合同");
  setText("#summary-next-action", nextAction);
  setText("#contract-parser-meta", state.extraction?.selected_sheet ? `${state.extraction.selected_sheet} · 表头 ${state.extraction.header_row} · 合同商品区段 ${contractItemRegionLabel()}` : "等待项目");
  const rows = $("#contract-items");
  rows.replaceChildren();
  if (!state.items.length) {
    rows.append(create("div", "empty-inline", "解析成功后显示全部货号、条码、四国品名和装箱数。"));
    return;
  }
  const head = create("div", "contract-item-head");
  ["货号", "条码", "中文品名", "意大利语品名", "西班牙语品名", "英语品名", "装箱数"].forEach((label) => head.append(create("span", "", label)));
  rows.append(head);
  const fragment = document.createDocumentFragment();
  state.items.forEach((item) => {
    const row = create("button", "contract-item-row");
    row.type = "button";
    row.append(
      create("strong", "", item.item_no || "—"),
      create("span", "", item.barcode || "无条码"),
      create("span", "", item.cn_name || "—"),
      create("span", "", item.it_name || "—"),
      create("span", "", item.es_name || "—"),
      create("span", "", item.en_name || "—"),
      create("span", "", item.qty == null ? "—" : `${item.qty} PCS`),
    );
    row.addEventListener("click", () => { state.selectedItemNo = item.item_no; showWorkspace("lineart"); renderSelection(); });
    fragment.append(row);
  });
  rows.append(fragment);
}

function setProgressSteps(done = [], active = null) {
  $$("#progress-steps li").forEach((node) => {
    const name = node.dataset.progressStep;
    node.classList.toggle("done", done.includes(name));
    node.classList.toggle("active", name === active);
    const label = node.querySelector("small");
    label.textContent = done.includes(name) ? "完成" : name === active ? "处理中" : "等待";
  });
}

function renderProgress() {
  const progress = state.progress;
  if (state.intake?.status === "uploading") {
    setText("#progress-title", "正在读取合同事实");
    setStatus("#progress-status", "解析中", "processing");
    $("#progress-fill").style.width = "18%";
    setText("#progress-primary", `读取 ${state.intake.filename}`);
    setText("#progress-secondary", "此阶段不伪造逐货号百分比；完成后立即显示产品区域的真实货号进度。");
    setProgressSteps([], "contract");
    return;
  }
  if (!state.projectId) {
    setText("#progress-title", "等待合同");
    setStatus("#progress-status", "未开始", "neutral");
    $("#progress-fill").style.width = "0%";
    setText("#progress-primary", "上传合同后显示真实处理阶段");
    setText("#progress-secondary", "合同读取与产品区域处理分别显示，不混成一个虚假百分比。");
    setProgressSteps([]);
    return;
  }
  const total = Number(progress?.total_items || state.items.length || 0);
  const completed = Number(progress?.completed_items || 0);
  if (["queued", "running"].includes(progress?.status)) {
    const percent = total ? Math.round((completed / total) * 100) : 0;
    setText("#progress-title", "产品区域逐货号处理中");
    setStatus("#progress-status", progress.status === "queued" ? "排队中" : "运行中", progress.status);
    $("#progress-fill").style.width = `${percent}%`;
    setText("#progress-primary", `${completed} / ${total} 个货号完成${progress.current_item_no ? ` · 当前 ${progress.current_item_no}` : ""}`);
    setText("#progress-secondary", "合同事实与原图已冻结；这里显示后台产品区域处理的持久化进度。");
    setProgressSteps(["contract", "images"], "regions");
    return;
  }
  if (progress?.status === "failed" || progress?.status === "interrupted") {
    setText("#progress-title", "产品区域处理未完成");
    setStatus("#progress-status", progress.status === "failed" ? "失败" : "已中断", progress.status);
    $("#progress-fill").style.width = `${total ? Math.round((completed / total) * 100) : 0}%`;
    setText("#progress-primary", `${completed} / ${total} 个货号完成`);
    setText("#progress-secondary", progress.error?.message || "可点击重新处理产品区域继续。");
    setProgressSteps(["contract", "images"], "regions");
    return;
  }
  const selected = state.items.filter((item) => item.product_region_status === "selected").length;
  const allSelected = state.items.length > 0 && selected === state.items.length;
  setText("#progress-title", allSelected ? "合同已进入线稿阶段" : "合同事实已解析");
  setStatus("#progress-status", allSelected ? "完成" : "待处理", allSelected ? "completed" : "needs_review");
  $("#progress-fill").style.width = allSelected ? "100%" : "46%";
  setText("#progress-primary", allSelected ? `${state.items.length} 个货号的产品区域已就绪` : `${selected} / ${state.items.length} 个产品区域已选择`);
  setText("#progress-secondary", allSelected ? "可以继续生成或整理线稿。" : "点击“处理产品区域”开始真实逐货号任务。");
  setProgressSteps(["contract", "images", ...(allSelected ? ["regions", "ready"] : [])], allSelected ? null : "regions");
}

function setAssetImage(imageSelector, emptySelector, buttonSelector, path, emptyText, alt) {
  const image = $(imageSelector);
  const empty = $(emptySelector);
  const button = $(buttonSelector);
  if (!path) {
    image.classList.add("hidden");
    clearImageSource(image);
    empty.classList.remove("hidden");
    empty.textContent = emptyText;
    button.disabled = true;
    button.dataset.imagePath = "";
    return;
  }
  image.classList.remove("hidden");
  empty.classList.add("hidden");
  image.alt = alt;
  // Frame shows a 960 px WebP (tens of KB); the lightbox opens the untouched original.
  setImageSource(image, thumbUrl(state.projectId, path, 960), {
    placeholder: thumbUrl(state.projectId, path, 160),
    onError: () => { image.classList.add("hidden"); empty.classList.remove("hidden"); empty.textContent = "图片文件不可读取"; },
  });
  button.disabled = false;
  button.dataset.imagePath = fileUrl(state.projectId, path);
  button.dataset.imageTitle = alt;
}

function markSelected(container, selector) {
  container.querySelectorAll(selector).forEach((node) => {
    node.classList.toggle("selected", String(node.dataset.itemNo) === String(state.selectedItemNo));
  });
}

function filteredWorkspaceItems() {
  const query = state.itemSearch.toLowerCase();
  if (!query) return state.items;
  return state.items.filter((item) => `${item.item_no || ""} ${item.cn_name || ""}`.toLowerCase().includes(query));
}

function renderLineartItemGrid() {
  const grid = $("#lineart-item-grid");
  const items = filteredWorkspaceItems();
  const ready = state.items.filter(isLineartReady).length;
  const eligible = state.items.filter((item) => item.product_region_status === "selected" && !isLineartReady(item)).length;
  setText("#lineart-item-summary", state.projectId ? `${items.length} 个可见 · ${ready}/${state.items.length} 就绪 · ${eligible} 个可批量生成` : "等待项目");
  if ($("#lineart-item-search").value !== state.itemSearch) $("#lineart-item-search").value = state.itemSearch;
  // Progress polls re-render this grid every second; rebuild only when content changed.
  const signature = JSON.stringify([state.projectId, items.map((item) => [item.item_no, item.source_image, item.cn_name, item.en_name, itemState(item).label])]);
  if (grid.dataset.signature === signature) {
    markSelected(grid, ".lineart-item-button");
    return;
  }
  grid.dataset.signature = signature;
  grid.replaceChildren();
  if (!items.length) {
    grid.append(create("div", "empty-inline", state.projectId ? "没有匹配的货号。" : "打开项目后显示全部货号。"));
    return;
  }
  const fragment = document.createDocumentFragment();
  items.forEach((item) => {
    const info = itemState(item);
    const button = create("button", `lineart-item-button${String(item.item_no) === String(state.selectedItemNo) ? " selected" : ""}`);
    button.type = "button";
    button.dataset.itemNo = item.item_no;
    if (item.source_image) {
      const image = create("img");
      image.src = thumbUrl(state.projectId, item.source_image, 160);
      image.alt = `${item.item_no} 产品图`;
      image.loading = "lazy";
      image.decoding = "async";
      button.append(image);
    } else {
      button.append(create("span", "lineart-item-thumb-empty", "缺图"));
    }
    const copy = create("div");
    copy.append(create("strong", "", item.item_no || "未编号"), create("small", "", item.cn_name || item.en_name || "未填写品名"));
    button.append(copy, create("span", `lineart-state-dot ${info.state}`));
    button.title = info.label;
    button.addEventListener("click", () => {
      state.selectedItemNo = item.item_no;
      const containing = allGroups().find((group) => group.members?.some((member) => String(member.item_no) === String(item.item_no)));
      state.selectedGroupId = containing?.group_id || null;
      renderSelection();
    });
    fragment.append(button);
  });
  grid.append(fragment);
}

function renderLineartWorkspace() {
  const item = getItem();
  const group = selectedGroup();
  const selectedRegion = item?.selected_product_region || item?.regionEntry?.selected_product_region || null;
  const regionPath = selectedRegion?.cutout_path || selectedRegion?.image_path || item?.regionEntry?.candidates?.[0]?.cutout_path || item?.regionEntry?.candidates?.[0]?.image_path;
  const lineartPath = item?.approved?.path || item?.lineart_asset || item?.candidates?.[0]?.path || null;
  const projectPromptInput = $("#project-lineart-prompt");
  const projectPrompt = state.projectPrompt?.prompt_instruction || "";
  if (document.activeElement !== projectPromptInput && projectPromptInput.value !== projectPrompt) {
    projectPromptInput.value = projectPrompt;
  }
  setText(
    "#project-prompt-status",
    state.projectPromptDirty
      ? "有未保存修改；生成前会自动保存"
      : projectPrompt
        ? "已用于本项目全部后续线稿生成"
        : "尚未设置，使用默认规则",
  );
  renderLineartItemGrid();
  setText("#lineart-view-title", group ? `${group.normalized_cn_name} · ${group.members?.length || 0} 个货号` : "产品区域与线稿");
  setText("#lineart-view-copy", group?.manual
    ? "人工类别只建立业务分组；选择代表线稿并确认同形后才会共享。"
    : group
      ? "从同类货号中选择产品，查看产品图、选区和入版线稿的完整关系。"
      : "直接从中央货号区选择产品；类别与序号只作为大合同的辅助导航。"
  );
  setAssetImage("#source-image", "#source-image-empty", "#source-image-button", item?.source_image, "暂无产品图片", item ? `${item.item_no} 合同产品图` : "产品图");
  setAssetImage("#region-image", "#region-image-empty", "#region-image-button", regionPath, item?.source_image ? "等待后台识别" : "缺产品图，无法自动选区", item ? `${item.item_no} 产品区域` : "产品区域");
  setAssetImage("#lineart-image", "#lineart-image-empty", "#lineart-image-button", lineartPath, "无线稿站位，请人工添加或生成", item ? `${item.item_no} 入版线稿` : "线稿");
  setText("#source-image-state", item?.source_image ? "合同原图" : "缺失");
  setText("#region-image-state", item?.product_region_status === "selected" ? "已选择" : item?.source_image ? "待处理" : "不可自动处理");
  renderRegionCandidates(item);
  setText("#lineart-image-state", isLineartReady(item) ? "已批准" : item?.lineart_status === "stale" ? "已失效" : "待处理");
  setText("#selected-item-number", item ? `货号 ${item.item_no}` : "未选择货号");
  setText("#selected-item-name", item?.cn_name || item?.en_name || "从左侧选择一个货号");
  const provenance = item?.lineart_provenance || {};
  const source = provenance.binding_scope === "category_group" ? `共享自 ${provenance.representative_item_no}` : provenance.provider || provenance.approval?.mode || "未建立";
  setText("#selected-item-meta", item ? `条码 ${item.barcode || "—"} · 装箱 ${item.qty ?? "—"} · 产品区域 ${item.product_region_status || "missing"} · 线稿来源 ${source}` : "这里会显示条码、装箱数、产品区域和线稿来源。");
  const promptInput = $("#lineart-prompt-instruction");
  const promptDraft = item ? state.promptDrafts[item.item_no] || "" : "";
  if (document.activeElement !== promptInput && promptInput.value !== promptDraft) promptInput.value = promptDraft;
  setHidden("#region-job-banner", !["queued", "running"].includes(state.progress?.status));
  if (["queued", "running"].includes(state.progress?.status)) {
    const total = Number(state.progress.total_items || 0);
    const completed = Number(state.progress.completed_items || 0);
    const percent = total ? Math.round((completed / total) * 100) : 0;
    setText("#region-job-title", "后台逐货号处理产品区域");
    setText("#region-job-copy", `${completed}/${total}${state.progress.current_item_no ? ` · 当前 ${state.progress.current_item_no}` : ""}`);
    $("#region-job-fill").style.width = `${percent}%`;
  }
  const lineartRunning = ["queued", "running"].includes(state.lineartProgress?.status);
  setHidden("#lineart-job-banner", !lineartRunning && !["failed", "interrupted", "completed"].includes(state.lineartProgress?.status));
  if (state.lineartProgress && state.lineartProgress.status !== "idle") {
    const total = Number(state.lineartProgress.total_items || 0);
    const completed = Number(state.lineartProgress.completed_items || 0);
    const succeeded = Number(state.lineartProgress.succeeded_items || 0);
    const failed = Number(state.lineartProgress.failed_items || 0);
    const percent = total ? Math.round((completed / total) * 100) : (lineartRunning ? 8 : 0);
    const fill = $("#lineart-job-fill");
    if (fill) fill.style.width = `${percent}%`;
    if (lineartRunning) {
      setText("#lineart-job-title", "后台生成缺失线稿");
      setText(
        "#lineart-job-copy",
        `${completed}/${total} 完成 · 成功 ${succeeded} · 失败 ${failed}${state.lineartProgress.current_item_no ? ` · 当前 ${state.lineartProgress.current_item_no}` : ""} · 并发 ${state.lineartProgress.concurrency || 2}`,
      );
    } else if (state.lineartProgress.status === "completed") {
      setText("#lineart-job-title", "线稿批量任务已完成");
      const skipped = (state.lineartProgress.skipped_items || []).length;
      const shareHint = onlyOptionalShareRemain()
        ? " · 其余货号未生成（共享可选；点「不共享，单独生成其余」或取消代表策略后重跑）"
        : "";
      setText("#lineart-job-copy", `成功 ${succeeded} · 失败 ${failed} · 跳过 ${skipped}${shareHint}`);
    } else if (state.lineartProgress.status === "failed" || state.lineartProgress.status === "interrupted") {
      setText("#lineart-job-title", "线稿批量任务未完成");
      setText("#lineart-job-copy", state.lineartProgress.error?.message || `已完成 ${completed}/${total}，可重新启动缺失项`);
    }
  }
  const shareable = Boolean(group && !group.direct && !group.sequence && (group.members?.length || 0) >= 2);
  setHidden("#share-panel", !shareable);
  if (!shareable) {
    setText("#share-title", "选择一个类别");
    setText("#share-copy", "系统只提出类别建议。共用线稿必须明确确认外轮廓相同，并选择代表货号。");
  } else {
    const approved = (group.members || []).filter((member) => isLineartReady(getItem(member.item_no))).length;
    setText("#share-title", `${group.normalized_cn_name} · ${group.members.length} 个货号`);
    setText("#share-copy", `${group.manual ? "人工类别" : "精确中文品名建议"}；${approved} 个成员已有可作为代表的批准线稿。建组本身不会复制资产。`);
  }

  // Policy note only when the selected SKU has diagnostic warnings; no permanent feature brochure.
  const warnings = []
    .concat(item?.lineart_quality_warnings || [])
    .concat(item?.quality_warnings || [])
    .concat(item?.approved?.quality_warnings || [])
    .filter(Boolean);
  const note = $("#quality-gate-note");
  if (note) {
    if (warnings.length) {
      note.classList.remove("hidden");
      note.innerHTML = `<strong>质量诊断提示：</strong>${warnings.slice(0, 3).map((value) => String(value)).join("；")}`;
    } else {
      note.classList.add("hidden");
      note.innerHTML = "";
    }
  }
  const pending = state.pendingShares || state.lineartProgress?.pending_shares || [];
  state.pendingShares = pending;
  const canShare = pending.length > 0 && pending.some((row) => {
    const rep = getItem(row.representative_item_no);
    return isLineartReady(rep);
  });
  setHidden("#pending-share-bar", !canShare);
  if (canShare) {
    const remain = missingLineartItems().length;
    setText("#pending-share-title", `${pending.length} 个类别可确认共享（可选）`);
    setText(
      "#pending-share-copy",
      remain
        ? `代表已就绪。共享可省 API；不共享也能生产——请点「不共享，单独生成其余」或取消勾选「只生成代表」后批量，或选中货号点「生成当前线稿」。仍缺 ${remain} 个入版线稿。`
        : "代表货号线稿已就绪；确认同形后一次绑定全组成员，不再调用图像 API。",
    );
    setDisabled("#generate-remaining-without-share", state.busy || !remain || ["queued", "running"].includes(state.lineartProgress?.status));
  }
}

const MARK_THUMB_WIDTH = 720;

function markThumbUrl(itemNo) {
  return markPreviewUrl(itemNo, { format: "webp", width: MARK_THUMB_WIDTH });
}

/** Gallery thumbnail first (usually cached already), then the lossless full render. */
function showMarkPreview(image, itemNo, onError) {
  setImageSource(image, markPreviewUrl(itemNo, { format: "webp" }), {
    placeholder: markThumbUrl(itemNo),
    onError,
  });
}

function renderMarkWorkspace() {
  const item = getItem();
  const hasPreview = Boolean(item && isLineartReady(item));
  setText("#mark-title", item ? `${item.item_no} · ${item.cn_name || item.en_name || "未填写品名"}` : "等待项目");
  setStatus("#mark-status", state.manifest?.status === "verified_pass" ? "已核对" : state.manifest?.status ? "已生成" : hasPreview ? "可生成" : "未生成", state.manifest?.status || (hasPreview ? "ready" : "neutral"));
  const image = $("#mark-image");
  const empty = $("#mark-image-empty");
  if (hasPreview) {
    image.classList.remove("hidden");
    empty.classList.add("hidden");
    showMarkPreview(image, item.item_no, () => { image.classList.add("hidden"); empty.classList.remove("hidden"); empty.textContent = "当前货号尚不能渲染正式唛头"; });
    $("#mark-image-button").disabled = false;
    $("#mark-image-button").dataset.imagePath = markPreviewUrl(item.item_no);
    $("#mark-image-button").dataset.imageTitle = `${item.item_no} 正式唛头`;
  } else {
    image.classList.add("hidden");
    clearImageSource(image);
    empty.classList.remove("hidden");
    empty.textContent = item ? "该货号线稿尚未达到入版门禁" : "打开项目后显示正式唛头";
    $("#mark-image-button").disabled = true;
  }
  setText("#mark-fact-item", item?.item_no || "—");
  setText("#mark-fact-name", item?.cn_name || item?.en_name || "—");
  setText("#mark-fact-barcode", item?.barcode || "—");
  setText("#mark-fact-qty", item?.qty ?? "—");
  setText("#mark-fact-lineart", item?.lineart_provenance?.binding_scope === "category_group" ? `共享 ${item.lineart_provenance.representative_item_no}` : item?.lineart_provenance?.provider || "—");
  setText("#delivery-file-name", state.projectId ? deliveryFilename() : "—");
  const strip = $("#mark-item-strip");
  setText("#mark-gallery-summary", state.projectId ? `${state.items.filter(isLineartReady).length}/${state.items.length} 个可预览 · 点击任一张放大到上方` : "等待项目");
  // Selecting a SKU only moves the highlight; thumbnails are kept, never re-requested.
  const signature = JSON.stringify([state.projectId, markPreviewUrl("_"), state.items.map((entry) => [entry.item_no, entry.cn_name, entry.en_name, isLineartReady(entry), itemState(entry).label])]);
  if (strip.dataset.signature === signature) {
    markSelected(strip, ".mark-gallery-item");
    return;
  }
  strip.dataset.signature = signature;
  strip.replaceChildren();
  if (!state.items.length) {
    strip.append(create("div", "empty-inline", "打开项目后显示货号。"));
    return;
  }
  state.items.forEach((entry) => {
    const ready = isLineartReady(entry);
    const button = create("button", `mark-gallery-item${String(entry.item_no) === String(state.selectedItemNo) ? " selected" : ""}`);
    button.type = "button";
    button.dataset.itemNo = entry.item_no;
    const thumb = ready ? create("img") : null;
    if (thumb) {
      thumb.alt = `${entry.item_no} 正式唛头预览`;
      thumb.src = markThumbUrl(entry.item_no);
      thumb.loading = "lazy";
      thumb.decoding = "async";
      thumb.onerror = () => thumb.replaceWith(create("div", "mark-gallery-empty", "正式唛头尚未生成"));
    } else {
      button.append(create("div", "mark-gallery-empty", entry.source_image ? "线稿未过入版门禁" : "缺产品图或手工线稿"));
    }
    const copy = create("div", "mark-gallery-copy");
    copy.append(create("strong", "", entry.item_no), create("small", "", entry.cn_name || entry.en_name || "未填写品名"), create("span", "", itemState(entry).label));
    if (thumb) button.append(thumb);
    button.append(copy);
    button.addEventListener("click", () => { state.selectedItemNo = entry.item_no; renderSelection(); });
    strip.append(button);
  });
}

function renderVerificationWorkspace() {
  const job = state.verifyProgress;
  const jobActive = ["queued", "running"].includes(job?.status);
  if (job && (jobActive || ["failed", "interrupted", "completed"].includes(job.status))) {
    const label = job.phase_label || job.phase || job.status || "核对";
    setText("#verify-job-title", jobActive ? "后台独立核对进行中" : (job.status === "completed" ? "后台独立核对完成" : "后台独立核对未完成"));
    setText(
      "#verify-job-copy",
      jobActive
        ? `${label}${job.total_items ? ` · ${job.completed_items || 0}/${job.total_items}` : ""}`
        : (job.error?.message || label),
    );
    const ratio = (job.completed_items || 0) / Math.max(job.total_items || 1, 1);
    const fill = jobActive ? Math.max(12, Math.min(95, Math.round(ratio * 100))) : (job.status === "completed" ? 100 : 0);
    const el = $("#verify-job-fill");
    if (el) el.style.width = `${fill}%`;
    setHidden("#verify-job-banner", false);
  } else {
    setHidden("#verify-job-banner", true);
  }
  const verification = projectVerification();
  const readiness = state.project?.readiness || {};
  const manualRelease = readiness.manual_release || {};
  const manualReleaseActive = readiness.release_mode === "manual_override" && manualRelease.active;
  const reportResults = state.verificationReport?.results || [];
  const total = verification?.total_items ?? state.items.length;
  const passed = verification?.passed_items ?? reportResults.filter((entry) => entry.review?.reviewer_status === "OK").length;
  const weak = verification?.weak_evidence_items ?? 0;
  setText("#verification-title", verification ? `${contractStem()} · 最新核对` : state.manifest ? "等待核对正式唛头" : "等待正式唛头");
  setStatus("#verification-status", manualReleaseActive ? "机器未通过 · 人工已放行" : verification?.status === "verified_pass" ? "核对通过" : verification?.status === "verified_failed" ? "核对发现问题" : "未执行", manualReleaseActive ? "manual_release" : verification?.status || "neutral");
  setText("#verification-passed", verification || reportResults.length ? `${passed}/${total || reportResults.length}` : "—");
  setText("#verification-weak", verification ? weak : "—");
  setText("#verification-coverage", verification ? (verification.coverage_complete ? "完整" : "不完整") : "—");

  const rows = buildVerificationListRows();
  if (!state.selectedVerificationItemNo || !rows.some((row) => String(row.item_no) === String(state.selectedVerificationItemNo))) {
    state.selectedVerificationItemNo = rows[0]?.item_no || state.selectedItemNo || null;
  }
  renderVerificationItemList(rows);
  renderVerificationDetail(state.selectedVerificationItemNo);

  const ready = Boolean(readiness.production_ready);
  const link = $("#download-pdf");
  if (link) {
    link.href = ready ? `/api/projects/${encodeURIComponent(state.projectId)}/files/generated/standard_mark.pdf` : "#";
    link.classList.toggle("disabled", !ready);
    link.setAttribute("aria-disabled", String(!ready));
  }
  const bar = $("#release-bar");
  if (bar) bar.dataset.state = ready ? "ready" : "locked";
  setText("#release-title", manualReleaseActive ? "人工确认后可交付" : ready ? "可交付" : "交付门禁锁定");
  setText("#release-copy", manualReleaseActive
    ? `${deliveryFilename()} 已由 ${manualRelease.release?.reviewer || "人工复核者"} 在保留机器失败结论的前提下确认放行；输入变化后自动失效。`
    : ready ? `${deliveryFilename()} 可交付` : (manualRelease.blocked_message || "只有最新核对完整通过，或完成当前问题的人工审核，且输入指纹未变化时才可下载。"));
  setText("#delivery-file-name", deliveryFilename());
  renderManualReleaseButton(readiness);
}

function renderManualReleaseButton(readiness = state.project?.readiness || {}) {
  const button = $("#open-manual-release");
  if (!button) return;
  const manual = readiness.manual_release || {};
  const active = readiness.release_mode === "manual_override" && manual.active;
  if (active) {
    button.disabled = false;
    button.classList.remove("hidden");
    button.textContent = "查看人工放行记录";
    button.title = "查看机器失败结论、逐项人工确认、操作人和放行理由";
    return;
  }
  const canRelease = manual.can_release === true;
  button.disabled = !canRelease;
  button.classList.toggle("hidden", readiness.stages?.verification?.automated_status !== "verified_failed");
  button.textContent = `人工确认异常并放行${manual.issue_count ? `（${manual.issue_count} 项）` : ""}`;
  button.title = canRelease ? "逐项人工审核当前机器核对问题" : (manual.blocked_message || "当前不能人工放行");
}

function openManualRelease() {
  const manual = state.project?.readiness?.manual_release || {};
  const active = Boolean(manual.active && manual.release);
  const release = manual.release || {};
  const issues = active ? (release.confirmed_issues || []) : (manual.issue_preview || []);
  const list = $("#manual-release-issues");
  if (!list || !issues.length) return;
  list.replaceChildren();
  issues.forEach((issue) => {
    const label = create("label", "manual-release-issue");
    const checkbox = create("input");
    checkbox.type = "checkbox";
    checkbox.name = "manual-release-issue";
    checkbox.value = issue.issue_id || "";
    checkbox.required = !active;
    checkbox.checked = active;
    checkbox.disabled = active;
    const copy = create("div");
    const item = issue.item_number ? `货号 ${issue.item_number} · ` : "核对门禁 · ";
    const status = issueStatusLabels[issue.status] || issue.status;
    copy.append(create("strong", "", `${item}${fieldLabels[issue.field] || issue.field} · ${status}`));
    const values = [issue.contract_value && `合同：${issue.contract_value}`, issue.mark_value && `唛头：${issue.mark_value}`, issue.note].filter(Boolean).join("；");
    if (values) copy.append(create("small", "", values));
    label.append(checkbox, copy);
    list.append(label);
  });
  setText("#manual-release-title", active ? "人工放行审核记录" : "人工确认异常并放行");
  setText(
    "#manual-release-copy",
    active
      ? `机器核对仍为未通过。以下 ${issues.length} 项已由 ${release.reviewer || "人工复核者"} 确认，放行时间：${release.released_at || "—"}。`
      : "机器核对结果不会被改写。请逐项检查当前问题，确认后才会放行当前这一个核对 run 的 PDF。",
  );
  const reviewer = $("#manual-release-reviewer");
  if (reviewer) {
    reviewer.value = active ? (release.reviewer || "") : (window.localStorage.getItem("mark-canvas:manual-release-reviewer") || "");
    reviewer.readOnly = active;
  }
  const reason = $("#manual-release-reason");
  reason.value = active ? (release.reason || "") : "";
  reason.readOnly = active;
  const finalConfirm = $("#manual-release-confirm");
  finalConfirm.checked = active;
  finalConfirm.disabled = active;
  const submit = $("#manual-release-submit");
  if (submit) submit.classList.toggle("hidden", active);
  openModal("manual-release-modal");
}

async function saveManualRelease() {
  if (!state.projectId) return;
  if (state.project?.readiness?.manual_release?.active) return;
  const reviewer = $("#manual-release-reviewer").value.trim();
  const reason = $("#manual-release-reason").value.trim();
  const confirmedIssueIds = $$('input[name="manual-release-issue"]:checked').map((input) => input.value);
  if (!confirmedIssueIds.length) {
    showNotice("无法放行", "请逐项勾选当前核对列出的异常事实。", "warning");
    return;
  }
  window.localStorage.setItem("mark-canvas:manual-release-reviewer", reviewer);
  await runAction("人工确认放行", async () => {
    await request(`/api/projects/${encodeURIComponent(state.projectId)}/verification/manual-release`, {
      method: "POST",
      body: formData({
        reviewer,
        reason,
        confirm_all_issues: $("#manual-release-confirm").checked,
        confirmed_issue_ids: JSON.stringify(confirmedIssueIds),
      }),
    });
    closeModal("manual-release-modal");
    await loadProject(state.projectId, { quiet: true });
  }, "当前核对异常已记录人工确认，PDF 可交付").catch(() => {});
}

function buildVerificationListRows() {
  const reportByNo = new Map((state.verificationReport?.results || []).map((entry) => [String(entry.item_number), entry]));
  const rows = state.items.map((item) => {
    const result = reportByNo.get(String(item.item_no)) || null;
    const status = result?.review?.reviewer_status || (result ? "UNKNOWN" : "");
    const failed = status && status !== "OK" && status !== "PASS";
    const manualConfirmed = failed && manualReleaseIssues().some(
      (issue) => String(issue.item_number || "") === String(item.item_no || ""),
    );
    return {
      item_no: item.item_no,
      barcode: item.barcode || "—",
      qty: item.qty ?? "—",
      names: [item.cn_name, item.it_name, item.es_name, item.en_name].filter(Boolean).join(" / ") || "—",
      status,
      failed,
      manualConfirmed,
      problem: summarizeVerificationProblems(result),
      result,
      item,
    };
  });
  rows.sort((a, b) => Number(b.failed) - Number(a.failed) || String(a.item_no).localeCompare(String(b.item_no), "en"));
  return rows;
}

function summarizeVerificationProblems(result) {
  if (!result) return state.verificationReport ? "无本货号结果" : "未核对";
  const status = result.review?.reviewer_status;
  if (status === "OK" || status === "PASS") return "—";
  // Never treat non-blocking barcode plain-OCR gaps as list problems.
  const bad = (result.fields || []).filter((field) => isActionableFieldIssue(field));
  if (!bad.length) return result.review?.reason_summary || result.error || status || "未通过";
  return bad.slice(0, 3).map((field) => {
    const label = fieldLabels[field.field] || field.field;
    return `${label}${field.status ? `(${field.status})` : ""}`;
  }).join("；");
}

function renderVerificationItemList(rows) {
  const body = $("#verification-item-list");
  if (!body) return;
  body.replaceChildren();
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.className = "empty-inline";
    td.textContent = state.projectId ? "当前项目没有货号。" : "等待核对报告。";
    tr.append(td);
    body.append(tr);
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.failed) tr.classList.add("fail");
    if (String(row.item_no) === String(state.selectedVerificationItemNo)) tr.classList.add("selected");
    const statusClass = !row.status ? "neutral" : row.manualConfirmed ? "manual" : row.failed ? "bad" : "ok";
    const statusText = !row.status ? "未核对" : row.manualConfirmed ? "机审未过 · 人工通过" : row.failed ? "未过" : "通过";
    tr.innerHTML = `
      <td><span class="status-pill ${statusClass}">${statusText}</span></td>
      <td>${row.item_no || "—"}</td>
      <td>${row.barcode}</td>
      <td>${row.qty}</td>
      <td>${row.names}</td>
      <td>${row.problem}</td>
    `;
    tr.addEventListener("click", () => {
      state.selectedVerificationItemNo = row.item_no;
      state.selectedItemNo = row.item_no;
      renderVerificationWorkspace();
    });
    body.append(tr);
  });
}

function formatFieldValue(value) {
  const text = String(value ?? "").trim();
  return text || "—";
}

function fieldStatusLabel(status) {
  if (status === "PASS" || status === "OK") return "通过";
  if (status === "MISMATCH") return "不符";
  if (status === "MISSING") return "缺失";
  if (status === "NEEDS_REVIEW") return "待审";
  return status || "待核对";
}

function isFieldOk(field) {
  return field?.status === "PASS" || field?.status === "OK";
}

function isBarcodePlainOcrGap(field) {
  // Systemic non-blocking policy: scan matched contract, human-readable OCR under bars missing.
  if (!field || !isFieldOk(field)) return false;
  if (field.field !== "barcode_text" && field.field !== "barcode_scan_vs_text") return false;
  return !String(field.mark_value || "").trim();
}

function isActionableFieldIssue(field) {
  if (!field?.status || isFieldOk(field)) return false;
  return true;
}

function fieldValueLabels(fieldName) {
  // Most fields compare contract truth vs mark evidence.
  // barcode_scan_vs_text compares scan decode vs human-readable OCR under the bars.
  if (fieldName === "barcode_scan_vs_text") return { left: "扫码", right: "明文OCR" };
  if (fieldName === "barcode_text") return { left: "合同", right: "明文OCR" };
  if (fieldName === "barcode_scan") return { left: "合同", right: "扫码" };
  return { left: "合同", right: "唛头" };
}

function fieldPairLines(field) {
  if (field?.field === "line_art") return lineArtFieldLines(field);
  if (isBarcodePlainOcrGap(field)) {
    const code = formatFieldValue(field.contract_value);
    if (field.field === "barcode_scan_vs_text") {
      return [`扫码 ${code}`, "明文 OCR：未识别（不阻断）"];
    }
    return [`合同 ${code}`, "明文 OCR：未识别（不阻断，扫码已通过）"];
  }
  const labels = fieldValueLabels(field.field);
  const left = formatFieldValue(field.contract_value);
  const right = formatFieldValue(field.mark_value);
  if (left === right) return [left];
  return [`${labels.left} ${left}`, `${labels.right} ${right}`];
}

function lineArtFieldLines(field) {
  const source = String(field.source || "");
  const note = String(field.note || "");
  const left = String(field.contract_value || "");
  const right = String(field.mark_value || "");
  const blob = `${source} ${note} ${left} ${right}`;
  const scoreMatch = blob.match(/(?:fuzzy score|模糊分)\s*([0-9]+(?:\.[0-9]+)?)/i);
  const score = scoreMatch ? scoreMatch[1] : "";
  const sizeMatch = blob.match(/(\d{2,5})x(\d{2,5})/i);
  const size = sizeMatch ? `${sizeMatch[1]}×${sizeMatch[2]}` : "";

  // Current fuzzy path
  if (source === "line_art_fuzzy_shape" || /模糊分|fuzzy score|外形/i.test(blob)) {
    if (isFieldOk(field)) {
      return score
        ? [`外形模糊核对：通过（分 ${score}）`]
        : ["外形模糊核对：通过"];
    }
    if (field.status === "NEEDS_REVIEW") {
      return score
        ? [`外形模糊核对：待审（分 ${score}）`]
        : ["外形模糊核对：待审"];
    }
    return score
      ? [`外形模糊核对：不符（分 ${score}）`]
      : ["外形模糊核对：不符"];
  }

  // No product photo path
  if (source === "line_art_nonblank_no_source" || /无产品图|without product photo/i.test(blob)) {
    return size
      ? [`无产品图对照，仅非空检查通过（${size}）`]
      : ["无产品图对照，仅非空检查通过"];
  }

  // Legacy report before fuzzy shape existed
  if (
    source === "line_art_trusted_no_visual_oc"
    || /trusted by policy|信任已生成线稿|已关闭视觉 OC/i.test(blob)
  ) {
    return [
      size ? `旧报告：仅非空检查通过（${size}）` : "旧报告：仅非空检查通过",
      "请重新运行「项目核对」以启用外形模糊核对",
    ];
  }

  if (source === "line_art_deterministic_qc" || /空白|不可读|blank/i.test(blob)) {
    return [note || right || left || "线稿完整性检查未通过"];
  }

  if (note) return [note];
  if (left && right && left !== right) return [left, right];
  return [right || left || "线稿检查"];
}

function fieldNoteText(field) {
  if (isBarcodePlainOcrGap(field)) {
    return "辅证：条码下方人眼数字 OCR 未读到；主证据以扫码为准。";
  }
  if (field?.field === "line_art") {
    const source = String(field.source || "");
    const note = String(field.note || "").trim();
    if (source === "line_art_fuzzy_shape") return note;
    if (source === "line_art_nonblank_no_source") {
      return note || "缺产品图时无法做外形模糊核对。";
    }
    if (source === "line_art_trusted_no_visual_oc" || /信任已生成线稿|trusted by policy/i.test(note)) {
      return "这是启用模糊外形核对之前的旧结果；重新核对后会显示模糊分。";
    }
    return note;
  }
  return String(field?.note || "").trim();
}

function manualReleaseIssues() {
  const manual = state.project?.readiness?.manual_release || {};
  return manual.active ? (manual.release?.confirmed_issues || []) : [];
}

function isFieldManuallyConfirmed(itemNo, field) {
  if (!field || isFieldOk(field)) return false;
  return manualReleaseIssues().some(
    (issue) => String(issue.item_number || "") === String(itemNo || "")
      && String(issue.field || "") === String(field.field || "")
      && String(issue.status || "") === String(field.status || ""),
  );
}

function fieldRowClass(field, itemNo) {
  if (isFieldManuallyConfirmed(itemNo, field)) return "field-check field-check-manual";
  if (!isFieldOk(field)) return "field-check field-check-fail";
  if (isBarcodePlainOcrGap(field)) return "field-check field-check-soft";
  return "field-check field-check-pass";
}

function fieldStatusChip(field, itemNo) {
  if (isFieldManuallyConfirmed(itemNo, field)) return { className: "manual", text: "机审未过 · 人工已确认" };
  if (isBarcodePlainOcrGap(field)) return { className: "soft", text: "通过·辅" };
  if (isFieldOk(field)) return { className: "ok", text: "通过" };
  return { className: "failed", text: fieldStatusLabel(field.status) };
}

function renderVerificationDetail(itemNo) {
  const item = getItem(itemNo);
  const result = currentVerificationResult(itemNo);
  const image = $("#verification-image");
  const empty = $("#verification-image-empty");
  const button = $("#verification-image-button");
  const detailTitle = $("#verification-detail-title");
  const detailMeta = $("#verification-detail-meta");
  if (detailTitle) detailTitle.textContent = itemNo ? String(itemNo) : "未选择货号";
  if (item && state.manifest?.status) {
    image.classList.remove("hidden");
    empty.classList.add("hidden");
    showMarkPreview(image, item.item_no, () => { image.classList.add("hidden"); empty.classList.remove("hidden"); empty.textContent = "当前货号尚无核对预览图"; });
    if (button) {
      button.disabled = false;
      button.dataset.imagePath = markPreviewUrl(item.item_no);
      button.dataset.imageTitle = `${item.item_no} 核对唛头证据`;
    }
  } else {
    image.classList.add("hidden");
    clearImageSource(image);
    empty.classList.remove("hidden");
    empty.textContent = "点选上方货号查看证据";
    if (button) button.disabled = true;
  }
  const fields = $("#verification-fields");
  fields.replaceChildren();
  if (!itemNo) {
    if (detailMeta) detailMeta.textContent = "点选上方列表行查看";
    fields.append(create("div", "empty-inline", "等待核对报告或点选货号。"));
    return;
  }
  if (!result?.fields?.length) {
    if (detailMeta) detailMeta.textContent = result ? "无逐字段结果" : "尚未核对或无报告";
    fields.append(create("div", "empty-inline", result ? "无逐字段结果。" : "等待核对报告或点选货号。"));
    return;
  }
  const all = result.fields || [];
  const failed = all.filter((field) => isActionableFieldIssue(field));
  const passed = all.filter((field) => isFieldOk(field));
  const softGaps = passed.filter((field) => isBarcodePlainOcrGap(field));
  const hardPassed = passed.filter((field) => !isBarcodePlainOcrGap(field));
  // Show solid passes first; soft OCR-gap passes last so they do not look like defects.
  const orderedPassed = hardPassed.concat(softGaps);
  if (detailMeta) {
    if (failed.length) {
      detailMeta.textContent = softGaps.length
        ? `${failed.length} 项异常 · ${hardPassed.length} 项通过 · ${softGaps.length} 项明文OCR未识别(不阻断)`
        : `${failed.length} 项异常 · ${passed.length} 项通过`;
    } else if (softGaps.length) {
      detailMeta.textContent = `主证据通过 · ${softGaps.length} 项明文OCR未识别(不阻断)`;
    } else {
      detailMeta.textContent = `全部 ${passed.length} 项通过`;
    }
  }

  const appendFieldRow = (field) => {
    const row = create("div", fieldRowClass(field, itemNo));
    const label = fieldLabels[field.field] || field.field || "字段";
    const copy = create("div");
    copy.append(create("strong", "", label));
    fieldPairLines(field).forEach((line, index) => {
      const softRight = isBarcodePlainOcrGap(field) && index > 0;
      copy.append(create("small", softRight ? "field-soft-value" : (index > 0 ? "field-mark-value" : ""), line));
    });
    const note = fieldNoteText(field);
    if (note) copy.append(create("small", isFieldOk(field) ? "field-note-ok" : "field-note", note));
    const chip = fieldStatusChip(field, itemNo);
    row.append(copy, create("span", chip.className, chip.text));
    fields.append(row);
  };

  const show = failed.length ? failed : orderedPassed;
  show.forEach(appendFieldRow);
  if (failed.length && orderedPassed.length) {
    const more = create("button", "text-button field-show-passed", `另有 ${orderedPassed.length} 项已通过（含辅证），点击展开`);
    more.type = "button";
    more.addEventListener("click", () => {
      more.remove();
      orderedPassed.forEach(appendFieldRow);
    });
    fields.append(more);
  }
}

function renderRegionCandidates(item) {
  const list = $("#region-candidates");
  if (!list) return;
  list.replaceChildren();
  const candidates = (item?.regionEntry?.candidates || []).filter((candidate) => candidate && candidate.status !== "rejected");
  if (!item) {
    list.append(create("div", "empty-inline", "选择货号后显示产品区域候选。"));
    return;
  }
  if (!candidates.length) {
    list.append(create("div", "empty-inline", item.source_image ? "尚未识别产品区域候选。" : "无产品图，无法自动选区。"));
    return;
  }
  const selectedId = item.selected_product_region?.candidate_id || item.regionEntry?.selection?.candidate_id;
  candidates.slice(0, 8).forEach((candidate) => {
    const button = create("button", `region-candidate${candidate.candidate_id === selectedId || candidate.status === "selected" ? " selected" : ""}`);
    button.type = "button";
    const img = document.createElement("img");
    const candidatePath = candidate.cutout_path || candidate.image_path;
    img.src = thumbUrl(state.projectId, candidatePath, 320);
    img.alt = candidate.candidate_id || "产品区域候选";
    img.loading = "lazy";
    img.decoding = "async";
    const score = candidate.score != null ? Number(candidate.score).toFixed(2) : "—";
    button.append(img, create("strong", "", candidate.candidate_id || "候选"), create("small", "", `分 ${score}`));
    button.title = candidate.status === "selected" ? "当前选用" : "点击选用此候选";
    if (candidate.candidate_id !== selectedId) {
      button.addEventListener("click", () => selectRegionCandidate(item.item_no, candidate.candidate_id).catch(() => {}));
    } else {
      button.addEventListener("click", () => openImageLightbox(fileUrl(state.projectId, candidatePath), `${item.item_no} ${candidate.candidate_id}`, "当前选用的产品区域"));
    }
    list.append(button);
  });
}

async function selectRegionCandidate(itemNo, candidateId) {
  await runAction("选择产品区域", async () => {
    await request(
      `/api/projects/${encodeURIComponent(state.projectId)}/product-regions/${encodeURIComponent(itemNo)}/select`,
      { method: "POST", body: formData({ candidate_id: candidateId }) },
    );
    await loadProject(state.projectId, { quiet: true });
  }, `${itemNo} 已改用 ${candidateId}；旧线稿如存在将失效`);
}

async function refineItemLineart() {
  const item = getItem();
  if (!item) return;
  const prompt = ($("#refine-lineart-prompt")?.value || "").trim();
  if (!prompt) {
    showNotice("需要提示词", "基于当前线稿再生成时必须填写本次修改要求。", "warning");
    return;
  }
  await runAction("线稿二次修改", async () => {
    await request(
      `/api/projects/${encodeURIComponent(state.projectId)}/linearts/${encodeURIComponent(item.item_no)}/refine`,
      { method: "POST", body: formData({ prompt_instruction: prompt, provider: "" }) },
    );
    if ($("#refine-lineart-prompt")) $("#refine-lineart-prompt").value = "";
    await loadProject(state.projectId, { quiet: true });
  }, `${item.item_no} 已按提示词修改并直接采用`);
}


function renderExternalRun() {
  const run = state.externalRun;
  const total = run?.items?.length || 0;
  const ok = Number(run?.status_counts?.OK || 0);
  const review = Number(run?.status_counts?.NEEDS_MANUAL_REVIEW || 0);
  const blocked = Number(run?.status_counts?.BLOCKING_ISSUE || 0);
  setStatus("#external-status", blocked ? "存在阻断" : review ? "需要人工复核" : "核对完成", blocked ? "failed" : review ? "attention" : "ready");
  const result = $("#external-result");
  result.replaceChildren();
  result.append(create("strong", "", `Run ${run.run_id} · ${ok}/${total} 通过`));
  result.append(create("span", "", `需复核 ${review} · 阻断 ${blocked} · 结果独立归档`));
  if (run.downloads?.visual) {
    const link = create("a", "", "打开可视化核对报告");
    link.href = run.downloads.visual;
    link.target = "_blank";
    result.append(link);
  }
}

function taskData() {
  if (!state.projectId) return [{ state: "waiting", workspace: "contract", title: "上传并解析合同", copy: "建立冻结的合同事实与货号清单。", done: false }];
  const total = state.items.length;
  const selected = state.items.filter((item) => item.product_region_status === "selected").length;
  const approved = state.items.filter(isLineartReady).length;
  const standard = Boolean(state.manifest?.status);
  const verified = state.manifest?.status === "verified_pass" && Boolean(state.project?.readiness?.production_ready);
  const manualRelease = state.project?.readiness?.release_mode === "manual_override";
  const running = ["queued", "running"].includes(state.progress?.status);
  return [
    { state: "done", workspace: "contract", title: "合同事实已冻结", copy: `${total} 个货号 · 表头第 ${state.extraction?.header_row || "?"} 行`, done: true },
    { state: selected === total ? "done" : running ? "running" : "waiting", workspace: "lineart", title: selected === total ? "产品区域已就绪" : "处理产品区域", copy: `${selected}/${total}${state.progress?.current_item_no ? ` · 当前 ${state.progress.current_item_no}` : ""}`, done: selected === total },
    { state: approved === total ? "done" : "waiting", workspace: "lineart", title: approved === total ? "入版线稿已就绪" : "生成或补充线稿", copy: `${approved}/${total} 已生成可用线稿`, done: approved === total },
    { state: standard ? "done" : approved === total ? "waiting" : "blocked", workspace: "mark", title: standard ? "正式唛头已生成" : "生成正式唛头", copy: deliveryFilename(), done: standard },
    { state: verified || manualRelease ? "done" : standard ? "waiting" : "blocked", workspace: "verify", title: manualRelease ? "人工确认放行，可交付" : verified ? "核对通过，可交付" : "执行独立核对", copy: manualRelease ? "异常事实已留痕确认" : verified ? "只交付 PDF" : "未通过前不可下载最终产物", done: verified || manualRelease },
  ];
}

function renderTasks() {
  const tasks = taskData();
  const filtered = tasks.filter((task) => state.taskFilter === "all" || (state.taskFilter === "done" ? task.done : !task.done));
  setText("#task-count", tasks.filter((task) => !task.done).length);
  $$("[data-task-filter]").forEach((button) => button.classList.toggle("active", button.dataset.taskFilter === state.taskFilter));
  const list = $("#task-list");
  list.replaceChildren();
  if (!filtered.length) {
    list.append(create("div", "empty-inline", "当前筛选下没有任务。"));
    return;
  }
  filtered.forEach((task) => {
    const button = create("button", "task-card");
    button.type = "button";
    button.dataset.state = task.state;
    const stripe = create("span");
    const copy = create("div");
    copy.append(create("strong", "", task.title), create("small", "", task.copy), create("em", "", task.done ? "已完成" : task.state === "blocked" ? "等待前置条件" : task.state === "running" ? "自动处理中" : "待处理"));
    button.append(stripe, copy);
    button.addEventListener("click", () => showWorkspace(task.workspace));
    list.append(button);
  });
}

function renderActions() {
  const hasProject = Boolean(state.projectId);
  const item = getItem();
  const group = selectedGroup();
  const shareable = Boolean(group && !group.direct && !group.sequence && (group.members?.length || 0) >= 2);
  const groupApproved = (group?.members || []).some((member) => isLineartReady(getItem(member.item_no)));
  const running = ["queued", "running"].includes(state.progress?.status);
  const lineartRunning = ["queued", "running"].includes(state.lineartProgress?.status);
  const missing = missingLineartItems();
  const lineartEligible = missing.length;
  const shareOnlyRemain = onlyOptionalShareRemain();
  const hasSource = Boolean(item?.source_image);
  const regionSelected = item?.product_region_status === "selected";
  setDisabled("#open-create-group", state.busy || !hasProject || state.items.length < 2 || state.grouping !== "category");
  setDisabled("#generate-regions", state.busy || !hasProject || !state.items.length || running);
  setDisabled("#generate-linearts", state.busy || !hasProject || !lineartEligible || lineartRunning);
  setDisabled("#generate-remaining-without-share", state.busy || !hasProject || !lineartEligible || lineartRunning || !shareOnlyRemain);
  const batch = $("#generate-linearts");
  if (batch) {
    if (lineartRunning) {
      batch.textContent = `线稿生成中 ${state.lineartProgress.completed_items || 0}/${state.lineartProgress.total_items || 0}`;
    } else if (!lineartEligible) {
      batch.textContent = "缺失线稿已全部处理";
    } else if (shareOnlyRemain && usesRepStrategy()) {
      // Rep strategy would generate 0 for these — make the CTA honest and actionable.
      batch.textContent = `单独生成其余 ${lineartEligible} 个（不共享）`;
    } else if (usesRepStrategy()) {
      batch.textContent = `按代表策略生成缺失线稿（${lineartEligible} 个待入版）`;
    } else {
      batch.textContent = `批量生成 ${lineartEligible} 个缺失线稿`;
    }
  }
  setDisabled("#generate-item-region", state.busy || !item || !hasSource);
  setDisabled("#generate-item-lineart", state.busy || !item || !regionSelected || lineartRunning);
  setDisabled("#refine-item-lineart", state.busy || !item || !isLineartReady(item) || lineartRunning);
  setDisabled("#refine-lineart-prompt", state.busy || !item || !isLineartReady(item) || lineartRunning);
  setDisabled("#lineart-prompt-instruction", state.busy || !item || !regionSelected);
  setDisabled("#open-manual-crop", state.busy || !item || !hasSource);
  setDisabled("#project-lineart-prompt", state.busy || !hasProject);
  setDisabled("#save-project-lineart-prompt", state.busy || !hasProject || !state.projectPromptDirty);
  setDisabled("#replace-lineart", state.busy || !item || !regionSelected);
  // Dedicated path for missing product photos only; replacement uses replace-lineart.
  setDisabled("#manual-lineart", state.busy || !item || hasSource);
  setDisabled("#open-share", state.busy || !shareable || !groupApproved);
  const verifyBusy = ["queued", "running"].includes(state.verifyProgress?.status);
  setDisabled("#generate-standard", state.busy || verifyBusy || !hasProject || !state.items.length || !state.items.every(isLineartReady));
  setDisabled("#generate-and-verify", state.busy || verifyBusy || !hasProject || !state.items.length || !state.items.every(isLineartReady));
  setDisabled("#contract-primary-action", state.busy);
  setDisabled("#open-contract", state.busy);
  setDisabled("#open-project", state.busy);
}

function openImageLightbox(url, title, meta = "") {
  if (!url) return;
  $("#lightbox-image").src = url;
  setText("#lightbox-title", title || "查看图片");
  setText("#lightbox-meta", meta || "使用下方滑块缩放图片");
  setLightboxZoom(100);
  openModal("image-lightbox");
}

function setLightboxZoom(value, anchorEvent = null) {
  const range = $("#zoom-range");
  const stage = $("#lightbox-stage");
  const image = $("#lightbox-image");
  const previous = Number(range.value || 100);
  const next = Math.max(Number(range.min || 40), Math.min(Number(range.max || 300), Math.round(value)));
  let anchorX = 0.5;
  let anchorY = 0.5;
  if (anchorEvent && stage.clientWidth && stage.clientHeight) {
    const rect = stage.getBoundingClientRect();
    anchorX = (stage.scrollLeft + anchorEvent.clientX - rect.left) / Math.max(stage.scrollWidth, 1);
    anchorY = (stage.scrollTop + anchorEvent.clientY - rect.top) / Math.max(stage.scrollHeight, 1);
  }
  range.value = String(next);
  $("#zoom-value").textContent = `${next}%`;
  image.style.width = `${next}%`;
  if (anchorEvent && next !== previous) {
    requestAnimationFrame(() => {
      stage.scrollLeft = Math.max(0, anchorX * stage.scrollWidth - (anchorEvent.clientX - stage.getBoundingClientRect().left));
      stage.scrollTop = Math.max(0, anchorY * stage.scrollHeight - (anchorEvent.clientY - stage.getBoundingClientRect().top));
    });
  }
}

function openImageFromButton(button) {
  if (button.disabled || !button.dataset.imagePath) return;
  openImageLightbox(button.dataset.imagePath, button.dataset.imageTitle, "仅提供缩放查看，不修改源文件。");
}

async function uploadContract(file, projectId = "") {
  if (!file) return;
  state.intake = { status: "uploading", filename: file.name };
  state.workspace = "contract";
  renderProgress();
  const data = new FormData();
  data.append("contract", file);
  if (projectId) data.append("project_id", projectId);
  await runAction("合同解析", async () => {
    const result = await request("/api/projects/extract", { method: "POST", body: data });
    state.projectId = result.project_id;
    await loadProject(result.project_id, { quiet: true });
    closeModal("contract-modal");
    const job = await request(`/api/projects/${encodeURIComponent(result.project_id)}/product-regions/generate`, { method: "POST", body: formData({ force: false }) });
    state.progress = job;
    renderAll();
    beginProgressPolling();
    if (result.warnings?.length) showNotice("合同已解析，存在提示", result.warnings.map((entry) => entry.message || entry.code || entry).join("；"));
    return result;
  }, "合同事实已冻结，产品区域正在后台逐货号处理");
}

async function startRegionJob(force = false) {
  if (!state.projectId) return;
  await runAction("产品区域处理", async () => {
    state.progress = await request(`/api/projects/${encodeURIComponent(state.projectId)}/product-regions/generate`, { method: "POST", body: formData({ force }) });
    renderAll();
    beginProgressPolling();
  }, "后台任务已启动，可继续浏览其他工作区");
}

function beginProgressPolling() {
  window.clearTimeout(state.progressTimer);
  const poll = async () => {
    if (!state.projectId) return;
    state.progress = await optionalRequest(`/api/projects/${encodeURIComponent(state.projectId)}/product-regions/progress`, state.progress, { timeoutMs: POLL_TIMEOUT_MS });
    renderProgress();
    renderLineartWorkspace();
    renderTasks();
    renderActions();
    if (["queued", "running"].includes(state.progress?.status)) {
      state.progressTimer = window.setTimeout(poll, 900);
    } else {
      await loadProject(state.projectId, { quiet: true });
      if (state.progress?.status === "completed") notify("产品区域处理完成", `${state.progress.completed_items}/${state.progress.total_items} 个货号已处理`);
    }
  };
  state.progressTimer = window.setTimeout(poll, 300);
}

async function generateLinearts(itemNo = null) {
  if (!state.projectId) return;
  if (!itemNo && ["queued", "running"].includes(state.lineartProgress?.status)) {
    showNotice("线稿任务进行中", "已有后台批量任务，请等待完成或查看进度条。", "warning");
    return;
  }
  const title = itemNo ? `生成 ${itemNo} 线稿` : "批量生成缺失线稿";
  await runAction(title, async () => {
    if (state.projectPromptDirty) await saveProjectLineartPrompt({ quiet: true });
    if (itemNo) {
      const promptInstruction = state.promptDrafts[itemNo] || "";
      await request(
        `/api/projects/${encodeURIComponent(state.projectId)}/linearts/${encodeURIComponent(itemNo)}/generate`,
        { method: "POST", body: formData({ provider: "", prompt_instruction: promptInstruction }) },
      );
      await loadProject(state.projectId, { quiet: true });
      return;
    }
    // Batch: background job with bounded concurrency and durable progress.
    // group_representatives never generates share-targets once the rep is ready.
    // If user still has missing SKUs that are only share targets, force all_missing
    // so "不想共享" never blocks production.
    let strategy = usesRepStrategy() ? "group_representatives" : "all_missing";
    if (strategy === "group_representatives" && onlyOptionalShareRemain()) {
      strategy = "all_missing";
      const checkbox = $("#lineart-strategy-reps");
      if (checkbox) checkbox.checked = false;
      showNotice(
        "已改为逐货号生成",
        "代表策略不会给同组其余货号调图像 API。已切换为「全部缺失」单独生成，共享仍可选、非强制。",
        "info",
      );
    }
    state.lineartProgress = await request(
      `/api/projects/${encodeURIComponent(state.projectId)}/linearts/generate-job`,
      { method: "POST", body: formData({ provider: "", strategy }) },
    );
    state.pendingShares = state.lineartProgress.pending_shares || [];
    renderLineartWorkspace();
    renderActions();
    beginLineartProgressPolling();
  }, itemNo
    ? `${itemNo} 已使用默认规则${state.projectPrompt?.prompt_instruction ? "、项目统一要求" : ""}${state.promptDrafts[itemNo] ? "和当前货号补充" : ""}生成并直接采用`
    : "缺失线稿已提交后台批量生成；可在进度条查看当前货号");
}

function beginLineartProgressPolling() {
  window.clearTimeout(state.lineartProgressTimer);
  const poll = async () => {
    if (!state.projectId) return;
    state.lineartProgress = await optionalRequest(
      `/api/projects/${encodeURIComponent(state.projectId)}/linearts/generate-progress`,
      state.lineartProgress,
      { timeoutMs: POLL_TIMEOUT_MS },
    );
    renderLineartWorkspace();
    renderActions();
    renderHeader();
    if (["queued", "running"].includes(state.lineartProgress?.status)) {
      state.lineartProgressTimer = window.setTimeout(poll, 1200);
      return;
    }
    await loadProject(state.projectId, { quiet: true });
    if (state.lineartProgress?.status === "completed") {
      const succeeded = state.lineartProgress.succeeded_items || 0;
      const failed = state.lineartProgress.failed_items || 0;
      state.pendingShares = state.lineartProgress.pending_shares || state.pendingShares || [];
      renderLineartWorkspace();
      const shareCount = (state.pendingShares || []).length;
      notify(
        "线稿批量完成",
        shareCount
          ? `成功 ${succeeded} · 失败 ${failed} · ${shareCount} 个类别可确认共享（可选）。不想共享请点「不共享，单独生成其余」。`
          : `成功 ${succeeded} · 失败 ${failed}`,
      );
    } else if (state.lineartProgress?.status === "failed" || state.lineartProgress?.status === "interrupted") {
      showNotice("线稿批量未完成", state.lineartProgress.error?.message || "可重新启动缺失项", "error");
    }
  };
  state.lineartProgressTimer = window.setTimeout(poll, 400);
}

function beginVerifyProgressPolling() {
  window.clearTimeout(state.verifyProgressTimer);
  const poll = async () => {
    if (!state.projectId) return;
    state.verifyProgress = await optionalRequest(
      `/api/projects/${encodeURIComponent(state.projectId)}/verify/progress`,
      state.verifyProgress,
      { timeoutMs: POLL_TIMEOUT_MS },
    );
    renderVerificationWorkspace();
    renderActions();
    renderHeader();
    if (["queued", "running"].includes(state.verifyProgress?.status)) {
      state.verifyProgressTimer = window.setTimeout(poll, 1000);
      return;
    }
    await loadProject(state.projectId, { quiet: true });
    if (state.verifyProgress?.status === "completed") {
      const st = state.verifyProgress.result_status || state.verifyProgress.verification?.status || "completed";
      notify("独立核对完成", st === "verified_pass" ? "已通过，可检查交付门禁" : `结果：${st}`);
    } else if (["failed", "interrupted"].includes(state.verifyProgress?.status)) {
      showNotice("独立核对未完成", state.verifyProgress.error?.message || "可重新点击生成并核对", "error");
    }
  };
  state.verifyProgressTimer = window.setTimeout(poll, 300);
}

async function saveProjectLineartPrompt({ quiet = false } = {}) {
  if (!state.projectId) return;
  const payload = await request(
    `/api/projects/${encodeURIComponent(state.projectId)}/lineart-prompt`,
    {
      method: "POST",
      body: formData({ prompt_instruction: state.projectPrompt?.prompt_instruction || "" }),
    },
  );
  state.projectPrompt = payload;
  state.projectPromptDirty = false;
  renderLineartWorkspace();
  renderActions();
  if (!quiet) notify("项目统一要求已保存", payload.prompt_instruction || "已清空，后续使用默认规则");
}

async function uploadReplacementLineart(file) {
  const item = getItem();
  if (!file || !item) return;
  const data = new FormData();
  data.append("file", file);
  await runAction("替换线稿", async () => {
    await request(`/api/projects/${encodeURIComponent(state.projectId)}/linearts/${encodeURIComponent(item.item_no)}/upload`, { method: "POST", body: data });
    await loadProject(state.projectId, { quiet: true });
  }, `${item.item_no} 替换线稿已绑定并通过完整性检查`);
}

function drawManualCrop() {
  const canvas = $("#manual-crop-canvas");
  const image = manualCropState.image;
  if (!image || !canvas.width || !canvas.height) return;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  const rect = manualCropState.rect;
  if (!rect) return;
  context.fillStyle = "rgba(19, 38, 27, 0.52)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.save();
  context.beginPath();
  context.rect(rect.x, rect.y, rect.width, rect.height);
  context.clip();
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  context.restore();
  context.strokeStyle = "#2f9b68";
  context.lineWidth = 3;
  context.setLineDash([9, 5]);
  context.strokeRect(rect.x, rect.y, rect.width, rect.height);
  context.setLineDash([]);
}

function manualCropPoint(event) {
  const canvas = $("#manual-crop-canvas");
  const bounds = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width, (event.clientX - bounds.left) * (canvas.width / bounds.width))),
    y: Math.max(0, Math.min(canvas.height, (event.clientY - bounds.top) * (canvas.height / bounds.height))),
  };
}

function updateManualCropRect(point) {
  const start = manualCropState.dragStart;
  if (!start) return;
  manualCropState.rect = {
    x: Math.min(start.x, point.x),
    y: Math.min(start.y, point.y),
    width: Math.abs(point.x - start.x),
    height: Math.abs(point.y - start.y),
  };
  const rect = manualCropState.rect;
  const valid = rect.width >= 8 && rect.height >= 8;
  $("#save-manual-crop").disabled = !valid;
  setText("#manual-crop-size", valid ? `选区占原图 ${Math.round((rect.width / $("#manual-crop-canvas").width) * 100)}% × ${Math.round((rect.height / $("#manual-crop-canvas").height) * 100)}%` : "选区太小，请框选完整产品");
  drawManualCrop();
}

function openManualCrop() {
  const item = getItem();
  if (!item?.source_image) return;
  manualCropState.rect = null;
  manualCropState.dragStart = null;
  manualCropState.dragging = false;
  $("#save-manual-crop").disabled = true;
  setText("#manual-crop-title", `${item.item_no} · 人工框选产品区域`);
  setText("#manual-crop-copy", "拖动框选完整产品。像红色尺寸标注这类外部说明可留在选区内，再通过本次线稿追加要求明确去除；合同原图保持不变。");
  setText("#manual-crop-size", "尚未选择区域");
  $("#manual-crop-empty").classList.remove("hidden");
  const rememberedReviewer = window.localStorage.getItem("mark-canvas:manual-crop-reviewer") || "";
  $("#manual-crop-reviewer").value = rememberedReviewer;
  const image = new Image();
  manualCropState.image = image;
  image.onload = () => {
    const maxWidth = 900;
    const maxHeight = 520;
    const scale = Math.min(1, maxWidth / image.naturalWidth, maxHeight / image.naturalHeight);
    const canvas = $("#manual-crop-canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    $("#manual-crop-empty").classList.add("hidden");
    const selectionId = item.selected_product_region?.candidate_id;
    const candidate = item.regionEntry?.candidates?.find((value) => value.candidate_id === selectionId);
    const bbox = candidate?.bbox;
    if (bbox) {
      manualCropState.rect = {
        x: (bbox.x / image.naturalWidth) * canvas.width,
        y: (bbox.y / image.naturalHeight) * canvas.height,
        width: (bbox.width / image.naturalWidth) * canvas.width,
        height: (bbox.height / image.naturalHeight) * canvas.height,
      };
      $("#save-manual-crop").disabled = false;
      setText("#manual-crop-size", "已载入当前自动选区，可直接调整");
    }
    drawManualCrop();
  };
  image.onerror = () => {
    manualCropState.image = null;
    setText("#manual-crop-empty", "产品图读取失败，无法人工选区");
  };
  image.src = fileUrl(state.projectId, item.source_image);
  openModal("manual-crop-modal");
}

async function saveManualCrop() {
  const item = getItem();
  const canvas = $("#manual-crop-canvas");
  const rect = manualCropState.rect;
  if (!item || !rect || rect.width < 8 || rect.height < 8) return;
  const reviewer = $("#manual-crop-reviewer").value.trim();
  const reason = $("#manual-crop-reason").value.trim();
  window.localStorage.setItem("mark-canvas:manual-crop-reviewer", reviewer);
  await runAction("保存人工产品选区", async () => {
    await request(`/api/projects/${encodeURIComponent(state.projectId)}/product-regions/${encodeURIComponent(item.item_no)}/manual-crop`, {
      method: "POST",
      body: formData({
        x: rect.x / canvas.width,
        y: rect.y / canvas.height,
        width: rect.width / canvas.width,
        height: rect.height / canvas.height,
        reviewer,
        reason,
      }),
    });
    closeModal("manual-crop-modal");
    await loadProject(state.projectId, { quiet: true });
  }, `${item.item_no} 已改用人工选区；旧线稿已按输入变更规则失效`).catch(() => {});
}

async function generateStandard(verify) {
  if (!state.projectId) return;
  await runAction(verify ? "生成并核对" : "生成正式唛头", async () => {
    // PDF generation stays synchronous; verification runs as a background job.
    const suffix = verify ? "?verify=true" : "";
    const payload = await request(`/api/projects/${encodeURIComponent(state.projectId)}/generate-standard${suffix}`, { method: "POST" });
    await loadProject(state.projectId, { quiet: true });
    if (verify) {
      state.verifyProgress = payload.verification_job || await request(`/api/projects/${encodeURIComponent(state.projectId)}/verify`, { method: "POST" });
      showWorkspace("verify");
      renderVerificationWorkspace();
      beginVerifyProgressPolling();
      return;
    }
    showWorkspace("mark");
  }, verify ? "正式 PDF 已生成，独立核对已在后台启动" : "正式 PDF 已生成，尚未核对时不可交付");
}

function populateCreateGroupModal() {
  const list = $("#manual-group-members");
  list.replaceChildren();
  state.items.forEach((item) => {
    const label = create("label");
    const checkbox = create("input");
    checkbox.type = "checkbox";
    checkbox.name = "manual-group-member";
    checkbox.value = item.item_no;
    label.append(checkbox, create("strong", "", item.item_no), create("span", "", item.cn_name || item.en_name || "未填写品名"));
    list.append(label);
  });
}

function populateShareModal() {
  const group = selectedGroup();
  const select = $("#share-representative");
  select.replaceChildren();
  if (!group) return;
  const approved = (group.members || []).map((member) => getItem(member.item_no)).filter(isLineartReady);
  approved.forEach((item) => select.append(new Option(`${item.item_no} · ${item.cn_name || item.en_name || "已批准线稿"}`, item.item_no)));
  setText("#share-modal-copy", `${group.normalized_cn_name} · ${group.members.length} 个货号。共享会记录代表货号、成员摘要、操作人和原因。`);
}

function bindEvents() {
  $$(".workspace-tab").forEach((button) => button.addEventListener("click", () => showWorkspace(button.dataset.workspace)));
  $("#open-contract").addEventListener("click", () => openModal("contract-modal"));
  $("#contract-primary-action").addEventListener("click", () => openModal("contract-modal"));
  $("#open-provider-settings")?.addEventListener("click", () => openProviderSettings());
  $("#provider-select")?.addEventListener("change", () => {
    const status = state.providerStatus;
    const models = (status?.available_models && status.available_models[$("#provider-select").value]) || [];
    const modelSelect = $("#provider-model-select");
    if (modelSelect) {
      modelSelect.replaceChildren();
      [...new Set(models.filter(Boolean))].forEach((model) => {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        modelSelect.append(option);
      });
    }
  });
  $("#provider-select-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const provider = $("#provider-select").value;
    const custom = $("#provider-model-custom").value.trim();
    const model = custom || $("#provider-model-select").value;
    await runAction("保存模型选择", async () => {
      state.providerStatus = await request("/api/lineart-provider/select", {
        method: "POST",
        body: formData({ provider, model }),
      });
      renderProviderStatus();
      closeModal("provider-modal");
    }, `已切换到 ${provider} / ${model}`);
  });
  $("#provider-key-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const target = $("#provider-key-target").value;
    const apiKey = $("#provider-api-key").value.trim();
    const append = $("#provider-key-append").checked;
    const path = target === "openai_image" ? "/api/lineart-provider/openai-key" : "/api/lineart-provider/grok-key";
    await runAction("保存 API Key", async () => {
      state.providerStatus = await request(path, {
        method: "POST",
        body: formData({ api_key: apiKey, append }),
      });
      $("#provider-api-key").value = "";
      renderProviderStatus();
    }, append ? "Key 已追加到本机密钥池" : "Key 已覆盖保存到本机");
  });
  $("#open-pending-share")?.addEventListener("click", () => {
    populatePendingShareModal();
    openModal("pending-share-modal");
  });
  $("#generate-remaining-without-share")?.addEventListener("click", () => {
    const checkbox = $("#lineart-strategy-reps");
    if (checkbox) checkbox.checked = false;
    generateLinearts().catch(() => {});
  });
  $("#lineart-strategy-reps")?.addEventListener("change", () => {
    renderActions();
    renderLineartWorkspace();
  });
  $("#pending-share-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPendingShares().catch(() => {});
  });
  $("#open-project").addEventListener("click", () => openModal("project-modal"));
  $("#notice-close").addEventListener("click", closeNotice);
  $$('[data-close-modal]').forEach((button) => button.addEventListener("click", () => closeModal(button.dataset.closeModal)));
  $$(".modal-backdrop").forEach((backdrop) => backdrop.addEventListener("click", (event) => { if (event.target === backdrop) closeModal(backdrop.id); }));
  $("#lightbox-close").addEventListener("click", () => closeModal("image-lightbox"));
  $("#zoom-range").addEventListener("input", (event) => {
    setLightboxZoom(Number(event.target.value));
  });
  $("#lightbox-stage").addEventListener("wheel", (event) => {
    event.preventDefault();
    const step = event.deltaY < 0 ? 15 : -15;
    setLightboxZoom(Number($("#zoom-range").value || 100) + step, event);
  }, { passive: false });
  ["#source-image-button", "#region-image-button", "#lineart-image-button", "#mark-image-button", "#verification-image-button"].forEach((selector) => {
    $(selector).addEventListener("click", () => openImageFromButton($(selector)));
  });
  $("#group-by-category").addEventListener("click", () => { state.grouping = "category"; renderGroupTree(); renderActions(); });
  $("#group-by-sequence").addEventListener("click", () => { state.grouping = "sequence"; renderGroupTree(); renderActions(); });
  $("#item-search").addEventListener("input", (event) => { state.itemSearch = event.target.value.trim(); renderGroupTree(); renderLineartItemGrid(); });
  $("#lineart-item-search").addEventListener("input", (event) => {
    state.itemSearch = event.target.value.trim();
    $("#item-search").value = state.itemSearch;
    renderGroupTree();
    renderLineartItemGrid();
  });
  $$('[data-task-filter]').forEach((button) => button.addEventListener("click", () => { state.taskFilter = button.dataset.taskFilter; renderTasks(); }));
  $("#contract-dropzone").addEventListener("click", () => $("#quick-contract-input").click());
  $("#quick-contract-input").addEventListener("change", (event) => uploadContract(event.target.files[0]).catch(() => {}));
  $("#contract-dropzone").addEventListener("dragover", (event) => event.preventDefault());
  $("#contract-dropzone").addEventListener("drop", (event) => { event.preventDefault(); uploadContract(event.dataTransfer.files[0]).catch(() => {}); });
  $("#contract-input").addEventListener("change", (event) => setText("#contract-upload-name", event.target.files[0]?.name || "选择 .xls / .xlsx"));
  $("#contract-form").addEventListener("submit", (event) => {
    event.preventDefault();
    uploadContract($("#contract-input").files[0], $("#project-id-input").value.trim()).catch(() => {});
  });
  $("#project-form").addEventListener("submit", (event) => {
    event.preventDefault();
    closeModal("project-modal");
    loadProject($("#project-open-input").value.trim());
  });
  $("#open-golden-project").addEventListener("click", () => { closeModal("project-modal"); loadProject(GOLDEN_PROJECT_ID); });
  $("#generate-regions").addEventListener("click", () => startRegionJob(false).catch(() => {}));
  $("#generate-item-region").addEventListener("click", async () => {
    const item = getItem();
    if (!item) return;
    await runAction("重新识别产品区域", async () => {
      await request(`/api/projects/${encodeURIComponent(state.projectId)}/product-regions/${encodeURIComponent(item.item_no)}/generate`, { method: "POST", body: formData({ force: true }) });
      await loadProject(state.projectId, { quiet: true });
    }, `${item.item_no} 产品区域已重新识别`).catch(() => {});
  });
  $("#generate-linearts").addEventListener("click", () => generateLinearts().catch(() => {}));
  $("#generate-item-lineart").addEventListener("click", () => generateLinearts(state.selectedItemNo).catch(() => {}));
  $("#refine-item-lineart")?.addEventListener("click", () => refineItemLineart().catch(() => {}));
  $("#project-lineart-prompt").addEventListener("input", (event) => {
    state.projectPrompt = { ...(state.projectPrompt || {}), prompt_instruction: event.target.value };
    state.projectPromptDirty = true;
    setText("#project-prompt-status", "有未保存修改；生成前会自动保存");
    renderActions();
  });
  $("#save-project-lineart-prompt").addEventListener("click", () => {
    runAction("保存项目统一要求", () => saveProjectLineartPrompt({ quiet: true }), "项目统一要求已保存").catch(() => {});
  });
  $("#lineart-prompt-instruction").addEventListener("input", (event) => {
    if (state.selectedItemNo) state.promptDrafts[state.selectedItemNo] = event.target.value;
  });
  $("#open-manual-crop").addEventListener("click", openManualCrop);
  $("#reset-manual-crop").addEventListener("click", () => {
    manualCropState.rect = null;
    manualCropState.dragStart = null;
    $("#save-manual-crop").disabled = true;
    setText("#manual-crop-size", "尚未选择区域");
    drawManualCrop();
  });
  $("#manual-crop-canvas").addEventListener("pointerdown", (event) => {
    if (!manualCropState.image) return;
    manualCropState.dragging = true;
    manualCropState.dragStart = manualCropPoint(event);
    manualCropState.rect = { x: manualCropState.dragStart.x, y: manualCropState.dragStart.y, width: 0, height: 0 };
    event.currentTarget.setPointerCapture(event.pointerId);
    updateManualCropRect(manualCropState.dragStart);
  });
  $("#manual-crop-canvas").addEventListener("pointermove", (event) => {
    if (manualCropState.dragging) updateManualCropRect(manualCropPoint(event));
  });
  ["pointerup", "pointercancel"].forEach((name) => $("#manual-crop-canvas").addEventListener(name, (event) => {
    if (!manualCropState.dragging) return;
    updateManualCropRect(manualCropPoint(event));
    manualCropState.dragging = false;
    manualCropState.dragStart = null;
  }));
  $("#manual-crop-form").addEventListener("submit", (event) => {
    event.preventDefault();
    saveManualCrop().catch(() => {});
  });
  $("#replace-lineart").addEventListener("click", () => $("#replace-lineart-input").click());
  $("#replace-lineart-input").addEventListener("change", (event) => uploadReplacementLineart(event.target.files[0]).catch(() => {}));
  $("#manual-lineart").addEventListener("click", () => {
    const item = getItem();
    setText("#manual-lineart-copy", `${item?.item_no || "当前货号"} 缺产品图时可手工添加。必须确认货号身份，并记录操作人和原因。`);
    openModal("manual-lineart-modal");
  });
  $("#manual-lineart-input").addEventListener("change", (event) => setText("#manual-lineart-name", event.target.files[0]?.name || "选择 PNG / JPG / WEBP"));
  $("#manual-lineart-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const item = getItem();
    if (!item) return;
    const data = new FormData();
    data.append("file", $("#manual-lineart-input").files[0]);
    data.append("reviewer", $("#manual-reviewer").value.trim());
    data.append("reason", $("#manual-reason").value.trim());
    data.append("confirm_item_identity", String($("#manual-confirm-item").checked));
    await runAction("手工添加线稿", async () => {
      await request(`/api/projects/${encodeURIComponent(state.projectId)}/linearts/${encodeURIComponent(item.item_no)}/upload-without-product-image`, { method: "POST", body: data });
      closeModal("manual-lineart-modal");
      event.target.reset();
      await loadProject(state.projectId, { quiet: true });
    }, `${item.item_no} 手工线稿已绑定当前合同 snapshot`).catch(() => {});
  });
  $("#open-create-group").addEventListener("click", () => { populateCreateGroupModal(); openModal("create-group-modal"); });
  $("#create-group-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const members = $$('input[name="manual-group-member"]:checked').map((input) => input.value);
    if (members.length < 2) { showNotice("无法创建类别", "请至少选择 2 个货号。", "error"); return; }
    await runAction("创建人工类别", async () => {
      const result = await request(`/api/projects/${encodeURIComponent(state.projectId)}/lineart-groups`, {
        method: "POST",
        body: formData({ item_nos: members.join(","), group_name: $("#create-group-name").value.trim(), reviewer: $("#create-group-reviewer").value.trim(), reason: $("#create-group-reason").value.trim() }),
      });
      closeModal("create-group-modal");
      event.target.reset();
      state.selectedGroupId = result.group.group_id;
      state.expandedGroups.add(result.group.group_id);
      await loadProject(state.projectId, { quiet: true });
    }, "类别已建立；尚未共享任何线稿").catch(() => {});
  });
  $("#open-share").addEventListener("click", () => { populateShareModal(); openModal("share-modal"); });
  $("#share-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const group = selectedGroup();
    if (!group) return;
    const variations = $$('input[name="share-variation"]:checked').map((input) => input.value).join(",");
    await runAction("共享类别线稿", async () => {
      await request(`/api/projects/${encodeURIComponent(state.projectId)}/lineart-groups/${encodeURIComponent(group.group_id)}/share`, {
        method: "POST",
        body: formData({ representative_item_no: $("#share-representative").value, reviewer: $("#share-reviewer").value.trim(), reason: $("#share-reason").value.trim(), allowed_variations: variations, confirm_same_shape: $("#share-confirm-shape").checked }),
      });
      closeModal("share-modal");
      event.target.reset();
      await loadProject(state.projectId, { quiet: true });
    }, `${group.normalized_cn_name} 已记录代表线稿与人工证据`).catch(() => {});
  });
  $("#generate-standard").addEventListener("click", () => generateStandard(false).catch(() => {}));
  $("#generate-and-verify").addEventListener("click", () => generateStandard(true).catch(() => {}));
  $("#open-manual-release").addEventListener("click", openManualRelease);
  $("#manual-release-form").addEventListener("submit", (event) => {
    event.preventDefault();
    saveManualRelease();
  });
  $("#download-pdf").addEventListener("click", (event) => { if (event.currentTarget.getAttribute("aria-disabled") === "true") event.preventDefault(); });
  $("#external-contract-input").addEventListener("change", (event) => setText("#external-contract-name", event.target.files[0]?.name || "选择 .xls / .xlsx"));
  $("#external-marks-input").addEventListener("change", (event) => setText("#external-marks-name", event.target.files.length ? `${event.target.files.length} 个唛头文件` : "选择 PDF / CDR / 图片，可多选"));
  const externalPanel = $("#external-verify-panel");
  if (externalPanel) {
    const syncExternalBadge = () => setText("#external-status", externalPanel.open ? "已展开" : "收起");
    externalPanel.addEventListener("toggle", syncExternalBadge);
    syncExternalBadge();
  }
  $("#external-verification-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData();
    data.append("contract", $("#external-contract-input").files[0]);
    [...$("#external-marks-input").files].forEach((file) => data.append("marks", file));
    await runAction("独立核对", async () => {
      state.externalRun = await request("/api/verification-runs", { method: "POST", body: data });
      renderExternalRun();
    }, "外部合同与唛头已形成独立核对 run").catch(() => {});
  });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeAllModals(); });
}


async function loadProviderStatus() {
  try {
    state.providerStatus = await request("/api/lineart-provider");
  } catch {
    state.providerStatus = null;
  }
  renderProviderStatus();
}

function renderProviderStatus() {
  const status = state.providerStatus;
  if (!status) {
    setText("#provider-active-label", "未加载");
    setText("#provider-active-model", "—");
    setText("#provider-grok-keys", "0");
    setText("#provider-openai-keys", "0");
    return;
  }
  const active = status.active_provider || status.provider || "—";
  setText("#provider-active-label", active === "grok_image" ? "Grok Imagine" : active === "openai_image" ? "OpenAI / 兼容网关" : active);
  setText("#provider-active-model", status.model || "—");
  setText("#provider-grok-keys", String(status.grok_key_count || 0));
  setText("#provider-openai-keys", String(status.openai_key_count || 0));
  const select = $("#provider-select");
  const modelSelect = $("#provider-model-select");
  if (select) select.value = active === "openai_compatible" ? "openai_image" : active;
  if (modelSelect) {
    const models = (status.available_models && status.available_models[select?.value || active]) || [];
    const unique = [...new Set(models.filter(Boolean))];
    modelSelect.replaceChildren();
    unique.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      if (model === status.model) option.selected = true;
      modelSelect.append(option);
    });
  }
}

function openProviderSettings() {
  renderProviderStatus();
  openModal("provider-modal");
}

function populatePendingShareModal() {
  const list = $("#pending-share-list");
  if (!list) return;
  list.replaceChildren();
  const pending = state.pendingShares || [];
  if (!pending.length) {
    list.append(create("div", "empty-inline", "当前没有待共享类别。"));
    return;
  }
  pending.forEach((row) => {
    const repReady = isLineartReady(getItem(row.representative_item_no));
    const node = create("div", "pending-share-item");
    node.innerHTML = `<strong>${row.normalized_cn_name || row.group_id}</strong><br>代表 ${row.representative_item_no} ${repReady ? "已就绪" : "尚未生成"} · 将共享给 ${(row.share_target_item_nos || []).join(", ")}`;
    list.append(node);
  });
}

async function submitPendingShares() {
  const pending = state.pendingShares || [];
  const reviewer = $("#pending-share-reviewer").value.trim();
  const reason = $("#pending-share-reason").value.trim();
  const confirm = $("#pending-share-confirm").checked;
  if (!pending.length) return;
  await runAction("类别共享", async () => {
    let shared = 0;
    const remaining = [];
    for (const row of pending) {
      const rep = getItem(row.representative_item_no);
      if (!isLineartReady(rep)) {
        remaining.push(row);
        continue;
      }
      await request(
        `/api/projects/${encodeURIComponent(state.projectId)}/lineart-groups/${encodeURIComponent(row.group_id)}/share`,
        {
          method: "POST",
          body: formData({
            representative_item_no: row.representative_item_no,
            reviewer,
            reason,
            allowed_variations: "size,color,pattern",
            confirm_same_shape: confirm,
          }),
        },
      );
      shared += 1;
    }
    state.pendingShares = remaining;
    closeModal("pending-share-modal");
    await loadProject(state.projectId, { quiet: true });
    return { shared, remaining: remaining.length };
  }, "类别线稿共享完成");
}

function latencyLabel(ms) {
  if (ms < 150) return { text: "网络良好", level: "good" };
  if (ms < 600) return { text: "网络较慢", level: "slow" };
  return { text: "网络很慢", level: "poor" };
}

async function checkService() {
  const pill = $("#service-pill");
  try {
    const started = performance.now();
    const health = await request("/api/health", { timeoutMs: POLL_TIMEOUT_MS });
    const elapsed = Math.round(performance.now() - started);
    const ok = health.status === "ok";
    const latency = latencyLabel(elapsed);
    pill.textContent = ok ? `服务就绪 · ${latency.text}` : "服务状态异常";
    pill.title = `往返 ${elapsed} ms（含服务器处理）。慢网下图片先显示缩略图，再补清晰图。`;
    pill.dataset.state = ok ? "ready" : "error";
    pill.dataset.latency = ok ? latency.level : "";
  } catch {
    pill.textContent = "服务不可用";
    pill.dataset.state = "error";
    pill.dataset.latency = "";
  }
}

async function bootstrap() {
  bindEvents();
  renderAll();
  // Independent startup reads run in parallel rather than as three serial round trips.
  const lastProject = window.localStorage.getItem("mark-canvas:last-project");
  await Promise.all([
    checkService(),
    loadProviderStatus(),
    lastProject ? loadProject(lastProject, { quiet: true }) : null,
  ]);
}

bootstrap();
