#!/usr/bin/env python3
"""
Migrate master_results.csv to new column structure.

Changes:
1. Rename: avg_latency_seconds → avg_total_time
2. Add: avg_time_to_first_token (set to 0.0000 for historical data)
"""

import csv
import shutil
from datetime import datetime
from pathlib import Path


def migrate_csv(filepath: str, backup: bool = True) -> None:
    """Migrate CSV file to new column structure.

    Args:
        filepath: Path to master_results.csv
        backup: Whether to create a backup before migration
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return

    # Create backup
    if backup:
        backup_path = filepath.parent / f"{filepath.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy2(filepath, backup_path)
        print(f"✓ Backup created: {backup_path}")

    # Read existing data
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        old_fieldnames = reader.fieldnames

    if not rows:
        print("⚠️  CSV is empty, nothing to migrate")
        return

    print(f"✓ Read {len(rows)} rows from CSV")

    # Check if migration is needed
    if 'avg_total_time' in old_fieldnames:
        print("⚠️  Migration already applied (avg_total_time column exists)")
        return

    if 'avg_latency_seconds' not in old_fieldnames:
        print("❌ Cannot migrate: avg_latency_seconds column not found")
        return

    # New column order
    new_columns = [
        "model_name",
        "model_class",
        "task",
        "reasoning_state",
        "total_prompts",
        "successful_prompts",
        "failed_prompts",
        "total_duration_seconds",
        "avg_total_time",  # Renamed from avg_latency_seconds
        "avg_time_to_first_token",  # NEW
        "total_tokens",
        "total_prompt_tokens",
        "total_completion_tokens",
        "throughput_tokens_per_second",
        "gpu_energy_wh",
        "co2_emissions_g",
        "tokens_per_joule",
        "avg_energy_per_prompt_wh",
        "timestamp",
        "error_message",
    ]

    # Transform rows
    migrated_rows = []
    for row in rows:
        new_row = {}
        for col in new_columns:
            if col == 'avg_total_time':
                # Rename: avg_latency_seconds → avg_total_time
                new_row[col] = row.get('avg_latency_seconds', '0.0000')
            elif col == 'avg_time_to_first_token':
                # Add new column with default value (historical data doesn't have TTFT)
                new_row[col] = '0.0000'
            else:
                # Copy existing data
                new_row[col] = row.get(col, '')

        migrated_rows.append(new_row)

    # Write migrated data
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_columns)
        writer.writeheader()
        writer.writerows(migrated_rows)

    print(f"✓ Migrated {len(migrated_rows)} rows")
    print(f"✓ Updated columns:")
    print(f"  - Renamed: avg_latency_seconds → avg_total_time")
    print(f"  - Added: avg_time_to_first_token (set to 0.0000 for historical data)")
    print(f"✓ Migration complete: {filepath}")


def main():
    """Main migration script."""
    import argparse

    parser = argparse.ArgumentParser(description='Migrate master_results.csv to new column structure')
    parser.add_argument('csv_file', help='Path to master_results.csv')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')

    args = parser.parse_args()

    print("=" * 80)
    print("CSV Column Migration Script")
    print("=" * 80)
    print()

    migrate_csv(args.csv_file, backup=not args.no_backup)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
