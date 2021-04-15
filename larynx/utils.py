import gruut_ipa

# Allow ' for primary stress and , for secondary stress
# Allow : for elongation
_IPA_TRANSLATE = str.maketrans(
    "',:",
    "".join(
        [
            gruut_ipa.IPA.STRESS_PRIMARY.value,
            gruut_ipa.IPA.STRESS_SECONDARY.value,
            gruut_ipa.IPA.LONG,
        ]
    ),
)
