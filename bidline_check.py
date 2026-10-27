import pandas as pd
import numpy as np


class BidlinePipelineValidator:
    def __init__(self, initial_reg, initial_std, initial_spc):
        """
        STAGE 1: Initialization
        Anchors the original 4-3-4-4 working/off days for each TECH_ID based on the initial files.
        """
        # Combine all initial working days to anchor the pattern
        df_on = pd.concat([initial_reg, initial_std, initial_spc], ignore_index=True)
        df_on["SHIFT_DATE"] = pd.to_datetime(df_on["SHIFT_DATE"])

        self.min_date = df_on["SHIFT_DATE"].min()
        self.tech_id_patterns = (
            {}
        )  # Stores the set of ON days (modulo 15) for each TECH_ID

        base_pattern = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        valid_patterns = [
            set(np.where(np.roll(base_pattern, i) == 1)[0]) for i in range(15)
        ]

        for tech_id, group in df_on.groupby("TECH_ID"):
            days = (group["SHIFT_DATE"] - self.min_date).dt.days
            mod_days = set(days % 15)
            for p in valid_patterns:
                if mod_days.issubset(p):
                    self.tech_id_patterns[str(tech_id)] = p
                    break

    def is_original_working_day(self, tech_id, date):
        """Helper for O(1) checks against the original 4-3-4-4 pattern."""
        tech_key = str(tech_id)
        if tech_key not in self.tech_id_patterns:
            return False
        day_offset = (date - self.min_date).days % 15
        return day_offset in self.tech_id_patterns[tech_key]

    def stage2_training_checks(self, std_training_df, spc_training_df, flex_df):
        """
        STAGE 2: Post-Training Validation
        1. Checks if all trainings are on original working days.
        2. Checks if each TECH_ID has at most 1 training per day.
        3. Verifies Flex days still do not fall on original working days.
        4. Checks that no training shares a dispatcher-day with a flex day.
        """
        results = {
            'invalid_training_days': [],
            'duplicate_trainings': [],
            'invalid_flex_days': [],
            'training_flex_overlap': [],
        }
        
        # 1 & 2: Training Checks
        df_train = pd.concat([std_training_df, spc_training_df], ignore_index=True)
        if not df_train.empty:
            df_train["SHIFT_DATE"] = pd.to_datetime(df_train["SHIFT_DATE"])

            # Check duplicate trainings per day
            duplicates = df_train[
                df_train.duplicated(subset=["TECH_ID", "SHIFT_DATE"], keep=False)
            ]
            results["duplicate_trainings"] = list(duplicates["TECH_ID"].unique())

            # Check if training falls on original working day
            for tech_id, group in df_train.groupby("TECH_ID"):
                for date in group["SHIFT_DATE"]:
                    if not self.is_original_working_day(tech_id, date):
                        results["invalid_training_days"].append(tech_id)
                        break

        # 3: Flex Checks
        if not flex_df.empty:
            flex_df = flex_df.copy()
            flex_df["SHIFT_DATE"] = pd.to_datetime(flex_df["SHIFT_DATE"])
            for tech_id, group in flex_df.groupby("TECH_ID"):
                for date in group["SHIFT_DATE"]:
                    if self.is_original_working_day(tech_id, date):
                        results["invalid_flex_days"].append(tech_id)
                        break

            if not df_train.empty:
                training_keys = set(zip(
                    df_train['TECH_ID'].astype(str), df_train['SHIFT_DATE']
                ))
                flex_keys = set(zip(
                    flex_df['TECH_ID'].astype(str), flex_df['SHIFT_DATE']
                ))
                results['training_flex_overlap'] = sorted({
                    f"{tech}@{date.date()}"
                    for tech, date in training_keys & flex_keys
                })

        results['invalid_training_days'] = list(set(results['invalid_training_days']))
        results['invalid_flex_days'] = list(set(results['invalid_flex_days']))
        return results

    def stage3_vacation_checks(self, vacation_df, supply_regular_qualified):
        """
        STAGE 3: Vacations Validation
        1. Checks if vacations are assigned on a working day from Supply_Regular_qualified.
        2. Checks total vacations per day is between 25 and 29 across all workcenters/shifts.
        """
        results = {"invalid_vacation_source": [], "daily_count_violations": {}}

        if vacation_df.empty:
            return results

        vacation_df = vacation_df.copy()
        vacation_df["SHIFT_DATE"] = pd.to_datetime(vacation_df["SHIFT_DATE"])

        # 1. Vacations must come from Supply_Regular_qualified
        if not supply_regular_qualified.empty:
            supply_regular_qualified = supply_regular_qualified.copy()
            supply_regular_qualified["SHIFT_DATE"] = pd.to_datetime(
                supply_regular_qualified["SHIFT_DATE"]
            )

            # Create an O(1) lookup set of valid (TECH_ID, Date) pairs available for vacation
            valid_vacation_sources = set(
                zip(
                    supply_regular_qualified["TECH_ID"],
                    supply_regular_qualified["SHIFT_DATE"],
                )
            )

            for _, row in vacation_df.iterrows():
                if (row["TECH_ID"], row["SHIFT_DATE"]) not in valid_vacation_sources:
                    results["invalid_vacation_source"].append(
                        (row["TECH_ID"], row["SHIFT_DATE"])
                    )

        results["invalid_vacation_source"] = list(
            set(results["invalid_vacation_source"])
        )

        # 2. Total number of vacations on each day must be between 25 and 29
        daily_counts = vacation_df.groupby("SHIFT_DATE").size()
        for date, count in daily_counts.items():
            if not (25 <= count <= 29):
                # Store the date (as string) and the failing count
                results["daily_count_violations"][date.strftime("%Y-%m-%d")] = count

        return results

    def stage4_flex_checks(self, final_flex_df):
        """
        STAGE 4: Final Flex Validation
        1. Checks if flex days are on original OFF days (not working days).
        2. Checks if total flex days over entire period per TECH_ID <= 4.
        """
        results = {"invalid_flex_days": [], "max_count_violations": []}

        if final_flex_df.empty:
            return results

        final_flex_df = final_flex_df.copy()
        final_flex_df["SHIFT_DATE"] = pd.to_datetime(final_flex_df["SHIFT_DATE"])

        for tech_id, group in final_flex_df.groupby("TECH_ID"):
            # Check Max Counts (Total Flex Days <= 4)
            if len(group) > 4:
                results["max_count_violations"].append(tech_id)

            # Check if on original OFF days
            for date in group["SHIFT_DATE"]:
                # If it IS an original working day, it's an invalid flex assignment
                if self.is_original_working_day(tech_id, date):
                    results["invalid_flex_days"].append(tech_id)
                    break

        results["invalid_flex_days"] = list(set(results["invalid_flex_days"]))
        return results
