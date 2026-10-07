import argparse
import numpy as np
import os
import pandas as pd

class CoverageReporter:
    @staticmethod
    def evaluate(demand_path: str, supply_path: str) -> pd.DataFrame:
        if not os.path.exists(demand_path):
            raise FileNotFoundError(f"Demand file not found: {demand_path}")
        if not os.path.exists(supply_path):
            raise FileNotFoundError(f"Supply file not found: {supply_path}")

        demand = pd.read_csv(demand_path)
        supply = pd.read_csv(supply_path)

        # Standardize date format to YYYY-MM-DD for merging
        demand['SHIFT_DATE_KEY'] = pd.to_datetime(demand['SHIFT_DATE']).dt.strftime('%Y-%m-%d')
        supply['SHIFT_DATE_KEY'] = pd.to_datetime(supply['SHIFT_DATE']).dt.strftime('%Y-%m-%d')

        # Aggregate supply shift count per workcenter, shift, and date
        supply_agg = (
            supply.groupby(['WORK_CENTER_ID', 'SHIFT_TYPE', 'SHIFT_DATE_KEY'])
            .size()
            .reset_index(name='SUPPLY')
            .rename(columns={'WORK_CENTER_ID': 'WORK_CENTER'})
        )

        # Merge demand and supply
        report = pd.merge(
            demand, 
            supply_agg, 
            on=['WORK_CENTER', 'SHIFT_TYPE', 'SHIFT_DATE_KEY'], 
            how='left'
        )
        report['SUPPLY'] = report['SUPPLY'].fillna(0).astype(int)

        # Compute key metrics
        report['NET_COVERAGE'] =  np.where(report["SUPPLY"] < report["DEMAND"], report["SUPPLY"], report["DEMAND"])
        report['UNDER_COVERAGE'] = report.apply(lambda r: max(0, r['DEMAND'] - r['SUPPLY']), axis=1)
        report['OVER_COVERAGE'] = report.apply(lambda r: max(0, r['SUPPLY'] - r['DEMAND']), axis=1)
        
        # Clean up temporary keys
        report.drop(columns=['SHIFT_DATE_KEY'], inplace=True)

        return report

    @staticmethod
    def print_summary(report_df: pd.DataFrame, member_name: str = "Solution"):
        total_demand = report_df['DEMAND'].sum()
        total_supply = report_df['SUPPLY'].sum()
        total_under = report_df['UNDER_COVERAGE'].sum()
        total_over = report_df['OVER_COVERAGE'].sum()
        fill_rate = round((min(total_supply, total_demand) / total_demand) * 100, 2)

        print(f"\n================ COVERAGE SUMMARY: {member_name} ================")
        print(f"Total Demand Required : {total_demand:,}")
        print(f"Total Supply Provided : {total_supply:,}")
        print(f"Unmet Demand (Under)  : {total_under:,}")
        print(f"Surplus Supply (Over) : {total_over:,}")
        print(f"Overall Fill Rate (%) : {fill_rate}%")
        print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate workforce supply coverage against demand.")
    parser.add_argument("--demand", type=str, default="data/Demand.csv", help="Path to input Demand.csv")
    parser.add_argument("--supply", type=str, required=True, help="Path to generated Supply.csv")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save detailed report CSV")
    parser.add_argument("--name", type=str, default="Solution", help="Name of the team member/solution")

    args = parser.parse_args()

    report_df = CoverageReporter.evaluate(args.demand, args.supply)
    CoverageReporter.print_summary(report_df, args.name)

    if args.output:
        report_df.to_csv(args.output, index=False)
        print(f"Detailed coverage report saved to {args.output}")