#!/usr/bin/env python3
"""
IPEDS Field Decoder Utility

This module provides human-readable interpretations of IPEDS coded fields.
Use this to understand what the numeric codes in your IPEDS data mean.

Usage:
    from ipeds_decoder import IPEDSDecoder

    decoder = IPEDSDecoder()
    print(decoder.decode_control(1))  # "Public"
    print(decoder.decode_size(3))     # "Medium (3,000-9,999)"
"""


class IPEDSDecoder:
    """Decoder for IPEDS institutional classification codes."""

    def __init__(self):
        """Initialize all IPEDS code mappings."""

        # Institution Control/Ownership
        self.control_codes = {
            1: "Public",
            2: "Private nonprofit",
            3: "Private for-profit",
            -1: "Not applicable",
            -2: "Not applicable",
        }

        # Institutional Level
        self.level_codes = {
            1: "Four or more years",
            2: "At least 2 but less than 4 years",
            3: "Less than 2 years",
            -1: "Not applicable",
            -2: "Not applicable",
        }

        # Highest Level of Offering
        self.highest_offering_codes = {
            0: "Other",
            1: "Award of less than one academic year",
            2: "Award of at least one but less than two academic years",
            3: "Associate degree",
            4: "Award of at least two but less than four academic years",
            5: "Bachelor's degree",
            6: "Postbaccalaureate certificate",
            7: "Master's degree",
            8: "Post-master's certificate",
            9: "Doctor's degree",
            -1: "Not applicable",
            -2: "Not applicable",
        }

        # Institution Size (FTE enrollment)
        self.size_codes = {
            1: "Very small (under 1,000)",
            2: "Small (1,000-2,999)",
            3: "Medium (3,000-9,999)",
            4: "Large (10,000-19,999)",
            5: "Very large (20,000 and above)",
            -1: "Not reported",
            -2: "Not applicable",
        }

        # Carnegie Basic Classification (2021)
        self.carnegie_codes = {
            15: "R1: Doctoral Universities - Very High Research Activity",
            16: "R2: Doctoral Universities - High Research Activity",
            17: "D/PU: Doctoral/Professional Universities",
            18: "M1: Master's Colleges & Universities - Larger Programs",
            19: "M2: Master's Colleges & Universities - Medium Programs",
            20: "M3: Master's Colleges & Universities - Small Programs",
            21: "Bac/A&S: Baccalaureate Colleges - Arts & Sciences Focus",
            22: "Bac/Diverse: Baccalaureate Colleges - Diverse Fields",
            23: "Bac/Assoc: Baccalaureate/Associate's Colleges",
            24: "Assoc/HT-High Trad: Associate's High Transfer-High Traditional",
            25: "Assoc/HT-Mixed: Associate's High Transfer-Mixed Traditional/Nontraditional",
            26: "Assoc/HT-High Nontr: Associate's High Transfer-High Nontraditional",
            27: "Assoc/Mixed-High Trad: Associate's Mixed Transfer/Career & Technical-High Traditional",
            28: "Assoc/Mixed-Mixed: Associate's Mixed Transfer/Career & Technical-Mixed Traditional/Nontraditional",
            29: "Assoc/Mixed-High Nontr: Associate's Mixed Transfer/Career & Technical-High Nontraditional",
            30: "Assoc/HC&T-High Trad: Associate's High Career & Technical-High Traditional",
            31: "Assoc/HC&T-Mixed: Associate's High Career & Technical-Mixed Traditional/Nontraditional",
            32: "Assoc/HC&T-High Nontr: Associate's High Career & Technical-High Nontraditional",
            33: "Spec/2-yr-Health: Special Focus Two-Year: Health Professions & Other Fields",
            34: "Spec/2-yr-Tech: Special Focus Two-Year: Technical Professions",
            35: "Spec/4-yr-Faith: Special Focus Four-Year: Faith-Related Institutions",
            36: "Spec/4-yr-Medical: Special Focus Four-Year: Medical Schools & Medical Centers",
            37: "Spec/4-yr-Health: Special Focus Four-Year: Other Health Professions Schools",
            38: "Spec/4-yr-Engin: Special Focus Four-Year: Engineering Schools",
            39: "Spec/4-yr-Tech: Special Focus Four-Year: Other Technology-Related Schools",
            40: "Spec/4-yr-Bus: Special Focus Four-Year: Business & Management Schools",
            41: "Spec/4-yr-Arts: Special Focus Four-Year: Arts, Music & Design Schools",
            42: "Spec/4-yr-Law: Special Focus Four-Year: Law Schools",
            43: "Spec/4-yr-Other: Special Focus Four-Year: Other Special Focus Institutions",
            -1: "Not classified",
            -2: "Not classified",
        }

        # Yes/No fields (1=Yes, 2=No for most IPEDS fields)
        self.yes_no_codes = {
            1: "Yes",
            2: "No",
            -1: "Not applicable",
            -2: "Not applicable",
        }

        # Special designation fields
        self.designation_codes = {
            1: "Yes",
            2: "No",
            -1: "Not applicable",
            -2: "Not applicable",
        }

    def decode_control(self, code):
        """Decode institution control/ownership code."""
        return self.control_codes.get(code, f"Unknown code: {code}")

    def decode_level(self, code):
        """Decode institutional level code."""
        return self.level_codes.get(code, f"Unknown code: {code}")

    def decode_highest_offering(self, code):
        """Decode highest level of offering code."""
        return self.highest_offering_codes.get(code, f"Unknown code: {code}")

    def decode_size(self, code):
        """Decode institution size code."""
        return self.size_codes.get(code, f"Unknown code: {code}")

    def decode_carnegie(self, code):
        """Decode Carnegie Basic Classification code."""
        return self.carnegie_codes.get(code, f"Unknown code: {code}")

    def decode_yes_no(self, code):
        """Decode yes/no field (1=Yes, 2=No)."""
        return self.yes_no_codes.get(code, f"Unknown code: {code}")

    def decode_hbcu(self, code):
        """Decode HBCU designation."""
        return (
            "Historically Black College/University"
            if code == 1
            else "Not HBCU" if code == 2 else f"Unknown: {code}"
        )

    def decode_tribal(self, code):
        """Decode Tribal College designation."""
        return (
            "Tribal College"
            if code == 1
            else "Not Tribal College" if code == 2 else f"Unknown: {code}"
        )

    def decode_landgrant(self, code):
        """Decode Land Grant designation."""
        return (
            "Land Grant Institution"
            if code == 1
            else "Not Land Grant" if code == 2 else f"Unknown: {code}"
        )

    def get_field_info(self, field_name):
        """Get information about a specific IPEDS field."""
        field_descriptions = {
            "CONTROL": "Institution control/ownership (1=Public, 2=Private nonprofit, 3=Private for-profit)",
            "ICLEVEL": "Institutional level (1=4+ years, 2=2-4 years, 3=<2 years)",
            "HLOFFER": "Highest level of offering (degrees/certificates offered)",
            "INSTSIZE": "Institution size by FTE enrollment (1=Very small to 5=Very large)",
            "CCBASIC": "Carnegie Basic Classification (research activity and degree focus)",
            "HBCU": "Historically Black Colleges and Universities (1=Yes, 2=No)",
            "TRIBAL": "Tribal College designation (1=Yes, 2=No)",
            "LANDGRNT": "Land Grant Institution (1=Yes, 2=No)",
            "MEDICAL": "Has medical degree programs (1=Yes, 2=No)",
            "HOSPITAL": "Operates a hospital (1=Yes, 2=No)",
            "UGOFFER": "Offers undergraduate programs (1=Yes, 2=No)",
            "GROFFER": "Offers graduate programs (1=Yes, 2=No)",
        }
        return field_descriptions.get(
            field_name, f"No description available for {field_name}"
        )

    def decode_row(self, row_dict, fields_to_decode=None):
        """Decode multiple fields from a data row."""
        if fields_to_decode is None:
            fields_to_decode = [
                "CONTROL",
                "ICLEVEL",
                "INSTSIZE",
                "CCBASIC",
                "HBCU",
                "TRIBAL",
                "LANDGRNT",
            ]

        decoded = {}
        for field in fields_to_decode:
            if field in row_dict and row_dict[field] is not None:
                value = row_dict[field]
                if field == "CONTROL":
                    decoded[f"{field}_decoded"] = self.decode_control(value)
                elif field == "ICLEVEL":
                    decoded[f"{field}_decoded"] = self.decode_level(value)
                elif field == "INSTSIZE":
                    decoded[f"{field}_decoded"] = self.decode_size(value)
                elif field == "CCBASIC":
                    decoded[f"{field}_decoded"] = self.decode_carnegie(value)
                elif field == "HBCU":
                    decoded[f"{field}_decoded"] = self.decode_hbcu(value)
                elif field == "TRIBAL":
                    decoded[f"{field}_decoded"] = self.decode_tribal(value)
                elif field == "LANDGRNT":
                    decoded[f"{field}_decoded"] = self.decode_landgrant(value)
                else:
                    decoded[f"{field}_decoded"] = self.decode_yes_no(value)

        return decoded

    def generate_cheat_sheet(self, output_file="ipeds_field_guide.txt"):
        """Generate a comprehensive cheat sheet of all IPEDS field codes."""
        with open(output_file, "w") as f:
            f.write("IPEDS FIELD DECODER CHEAT SHEET\n")
            f.write("=" * 50 + "\n")
            f.write("Generated for University Search Application Development\n")
            f.write(
                "This guide explains what all the numeric codes in IPEDS data mean.\n\n"
            )

            # Institution Control
            f.write("🏛️ CONTROL - Institution Control/Ownership\n")
            f.write("-" * 45 + "\n")
            for code, meaning in self.control_codes.items():
                f.write(f"   {code} = {meaning}\n")
            f.write("   Use: control_type column (human-readable version)\n\n")

            # Institutional Level
            f.write("🎓 ICLEVEL - Institutional Level\n")
            f.write("-" * 35 + "\n")
            for code, meaning in self.level_codes.items():
                f.write(f"   {code} = {meaning}\n")
            f.write("   Use: institutional_level column (human-readable version)\n\n")

            # Highest Offering
            f.write("📜 HLOFFER - Highest Level of Offering\n")
            f.write("-" * 40 + "\n")
            for code, meaning in self.highest_offering_codes.items():
                f.write(f"   {code} = {meaning}\n")
            f.write("   Note: 9 is most common for universities (Doctor's degree)\n\n")

            # Institution Size
            f.write("📊 INSTSIZE - Institution Size (by FTE enrollment)\n")
            f.write("-" * 50 + "\n")
            for code, meaning in self.size_codes.items():
                f.write(f"   {code} = {meaning}\n")
            f.write("   Use: size_category column (human-readable version)\n\n")

            # Carnegie Classification
            f.write("🏆 CCBASIC - Carnegie Basic Classification\n")
            f.write("-" * 45 + "\n")
            f.write("Research Universities:\n")
            for code in [15, 16, 17]:
                if code in self.carnegie_codes:
                    f.write(f"   {code} = {self.carnegie_codes[code]}\n")

            f.write("\nMaster's Universities:\n")
            for code in [18, 19, 20]:
                if code in self.carnegie_codes:
                    f.write(f"   {code} = {self.carnegie_codes[code]}\n")

            f.write("\nBaccalaureate Colleges:\n")
            for code in [21, 22, 23]:
                if code in self.carnegie_codes:
                    f.write(f"   {code} = {self.carnegie_codes[code]}\n")

            f.write("\nAssociate's Colleges (Community Colleges):\n")
            for code in range(24, 33):
                if code in self.carnegie_codes:
                    f.write(f"   {code} = {self.carnegie_codes[code]}\n")

            f.write("\nSpecial Focus Institutions:\n")
            for code in range(33, 44):
                if code in self.carnegie_codes:
                    f.write(f"   {code} = {self.carnegie_codes[code]}\n")

            f.write("   Use: carnegie_basic_desc column (human-readable version)\n\n")

            # Special Designations
            f.write("🏛️ SPECIAL DESIGNATIONS (1=Yes, 2=No)\n")
            f.write("-" * 45 + "\n")
            special_fields = {
                "HBCU": "Historically Black Colleges and Universities",
                "TRIBAL": "Tribal College",
                "LANDGRNT": "Land Grant Institution",
                "MEDICAL": "Has Medical Degree Programs",
                "HOSPITAL": "Operates a Hospital",
            }

            for field, description in special_fields.items():
                f.write(f"{field}: {description}\n")
                f.write("   1 = Yes (has this designation)\n")
                f.write("   2 = No (does not have this designation)\n\n")

            # Program Offerings
            f.write("📚 PROGRAM OFFERINGS (1=Yes, 2=No)\n")
            f.write("-" * 40 + "\n")
            f.write("UGOFFER: Offers Undergraduate Programs\n")
            f.write("GROFFER: Offers Graduate Programs\n")
            f.write("DEGGRANT: Degree-Granting Institution\n")
            f.write("   1 = Yes, 2 = No\n\n")

            # Admissions Fields
            f.write("📝 ADMISSIONS DATA FIELDS\n")
            f.write("-" * 30 + "\n")
            f.write("APPLCN: Total applications received\n")
            f.write("ADMSSN: Total admissions offered\n")
            f.write("ENRLT: Total enrolled\n")
            f.write("acceptance_rate: Calculated percentage (ADMSSN/APPLCN * 100)\n")
            f.write("yield_rate: Calculated percentage (ENRLT/ADMSSN * 100)\n\n")

            # Test Scores
            f.write("📊 STANDARDIZED TEST SCORES\n")
            f.write("-" * 35 + "\n")
            f.write("SAT Scores:\n")
            f.write(
                "   SATVR25/SATVR75: Evidence-Based Reading & Writing (25th/75th percentile)\n"
            )
            f.write("   SATMT25/SATMT75: Math (25th/75th percentile)\n")
            f.write("   sat_total_25/sat_total_75: Combined scores (calculated)\n\n")
            f.write("ACT Scores:\n")
            f.write("   ACTCM25/ACTCM75: Composite (25th/75th percentile)\n")
            f.write("   ACTEN25/ACTEN75: English (25th/75th percentile)\n")
            f.write("   ACTMT25/ACTMT75: Math (25th/75th percentile)\n\n")

            # Enrollment
            f.write("👥 ENROLLMENT DATA\n")
            f.write("-" * 20 + "\n")
            f.write("EFTOTLT: Total fall enrollment\n")
            f.write("student_body_size: Cleaned enrollment number\n")
            f.write("enrollment_size_category: Human-readable size category\n\n")

            # Quick Reference
            f.write("🎯 QUICK REFERENCE FOR APP DEVELOPMENT\n")
            f.write("-" * 45 + "\n")
            f.write("USE THESE HUMAN-READABLE COLUMNS:\n")
            f.write("✅ control_type (instead of CONTROL)\n")
            f.write("✅ size_category (instead of INSTSIZE)\n")
            f.write("✅ institutional_level (instead of ICLEVEL)\n")
            f.write("✅ carnegie_basic_desc (instead of CCBASIC)\n")
            f.write("✅ location (instead of CITY + STABBR)\n")
            f.write("✅ selectivity_category (calculated from acceptance_rate)\n\n")

            f.write("MOST USEFUL SEARCH FIELDS:\n")
            f.write("🔍 INSTNM: Institution name\n")
            f.write("🔍 location: City, State\n")
            f.write("🔍 control_type: Public, Private nonprofit, Private for-profit\n")
            f.write("🔍 size_category: Very small, Small, Medium, Large, Very large\n")
            f.write("🔍 acceptance_rate: Percentage (lower = more selective)\n")
            f.write("🔍 student_body_size: Number of students\n")
            f.write("🔍 sat_total_75: SAT scores (75th percentile)\n")
            f.write("🔍 carnegie_basic_desc: Academic focus/research level\n\n")

            f.write("SEARCH FILTER SUGGESTIONS:\n")
            f.write("📍 State: Use STABBR (CA, NY, TX, etc.)\n")
            f.write("🏛️ Type: Use control_type\n")
            f.write("📏 Size: Use size_category or student_body_size\n")
            f.write("🎯 Selectivity: Use acceptance_rate or selectivity_category\n")
            f.write("📊 Test Scores: Use sat_total_75 or ACTCM75\n")
            f.write("🎓 Level: Use institutional_level\n\n")

            f.write("-" * 50 + "\n")
            f.write("Generated by IPEDS Decoder Utility\n")
            f.write("For technical questions, refer to IPEDS documentation\n")
            f.write("Data source: U.S. Department of Education, NCES, IPEDS\n")


def main():
    """Generate IPEDS cheat sheet and show example usage."""
    print("🔍 IPEDS Code Decoder - Generating Cheat Sheet")
    print("=" * 50)

    decoder = IPEDSDecoder()

    # Generate the cheat sheet
    cheat_sheet_file = "ipeds_field_guide.txt"
    decoder.generate_cheat_sheet(cheat_sheet_file)
    print(f"✅ Cheat sheet generated: {cheat_sheet_file}")

    # Show example usage
    print("\n📊 Example decoding:")
    print(f"CONTROL = 1: {decoder.decode_control(1)}")
    print(f"ICLEVEL = 1: {decoder.decode_level(1)}")
    print(f"INSTSIZE = 3: {decoder.decode_size(3)}")
    print(f"CCBASIC = 18: {decoder.decode_carnegie(18)}")

    print(f"\n📋 Quick field lookup:")
    for field in ["CONTROL", "ICLEVEL", "INSTSIZE"]:
        print(f"{field}: {decoder.get_field_info(field)}")

    print(f"\n📄 Open '{cheat_sheet_file}' for the complete reference guide!")


if __name__ == "__main__":
    main()
