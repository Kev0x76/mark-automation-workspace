from __future__ import annotations

import argparse
import json
from pathlib import Path

from .services.excel_contract import extract_contract_project
from .settings import workspace_root_from_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract contract data and product images.")
    parser.add_argument("contract", type=Path, help="Path to contract .xlsx or legacy .xls")
    parser.add_argument("--project-id", help="Project id to write under projects/")
    parser.add_argument("--workspace-root", type=Path, default=workspace_root_from_app())
    args = parser.parse_args()

    result = extract_contract_project(
        args.contract,
        workspace_root=args.workspace_root,
        project_id=args.project_id,
    )
    print(
        json.dumps(
            {
                "project_id": result["normalized"]["project_id"],
                "total_items": result["report"]["total_items"],
                "matched_images": result["report"]["matched_images"],
                "warnings": len(result["report"]["warnings"]),
                "paths": result["paths"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
