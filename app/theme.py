from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

# Minimal blue and white look for an academic demo: white cards with thin borders on a very light blue page,
# dark blue text and one blue accent (same blue as the 3D results, src/demo.py height_colors).
ACCENT = colors.Color(
    name="academic_blue",
    c50="#eff4fb", c100="#dce7f6", c200="#b9cfec", c300="#8cb0de", c400="#5b8acb", c500="#3a6db5",
    c600="#1d4f91", c700="#1a4279", c800="#183863", c900="#162f52", c950="#0e1d33",
)

PAGE, CARD, LINE, INK, MUTED = "#f4f7fb", "#fdfeff", "#dce3ec", "#17233a", "#5a6679"
PAGE_DARK, CARD_DARK, LINE_DARK, INK_DARK, MUTED_DARK = "#0f1520", "#161e2b", "#2a3546", "#e4e9f1", "#9aa6b8"


class LabTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=ACCENT,
            secondary_hue=ACCENT,
            neutral_hue=colors.slate,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_sm,
            text_size=sizes.text_md,
            font=(fonts.GoogleFont("IBM Plex Sans", weights=(400, 500, 600)), "ui-sans-serif", "system-ui",
                  "sans-serif"),
            font_mono=(fonts.GoogleFont("IBM Plex Mono", weights=(400, 500)), "ui-monospace", "monospace"),
        )
        self.name = "lab"
        super().set(
            body_background_fill=PAGE,
            body_background_fill_dark=PAGE_DARK,
            body_text_color=INK,
            body_text_color_dark=INK_DARK,
            body_text_color_subdued=MUTED,
            body_text_color_subdued_dark=MUTED_DARK,
            background_fill_primary=CARD,
            background_fill_primary_dark=CARD_DARK,
            background_fill_secondary=PAGE,
            background_fill_secondary_dark=PAGE_DARK,
            border_color_primary=LINE,
            border_color_primary_dark=LINE_DARK,
            block_background_fill=CARD,
            block_background_fill_dark=CARD_DARK,
            block_border_color=LINE,
            block_border_color_dark=LINE_DARK,
            block_shadow="none",
            block_label_background_fill="transparent",
            block_label_background_fill_dark="transparent",
            block_label_border_width="0px",
            block_label_text_color=MUTED,
            block_label_text_color_dark=MUTED_DARK,
            block_title_text_color=INK,
            block_title_text_color_dark=INK_DARK,
            block_title_text_weight="500",
            input_background_fill=CARD,
            input_background_fill_dark=CARD_DARK,
            input_border_color=LINE,
            input_border_color_dark=LINE_DARK,
            input_shadow_focus="0 0 0 2px *primary_200",
            input_shadow_focus_dark="0 0 0 2px *primary_800",
            button_primary_background_fill="*primary_600",
            button_primary_background_fill_hover="*primary_700",
            button_primary_background_fill_dark="*primary_400",
            button_primary_background_fill_hover_dark="*primary_300",
            button_primary_text_color=CARD,
            button_primary_text_color_dark=PAGE_DARK,
            button_primary_border_color="*primary_600",
            button_primary_border_color_dark="*primary_400",
            button_primary_shadow="none",
            button_secondary_background_fill=CARD,
            button_secondary_background_fill_hover=PAGE,
            button_secondary_background_fill_dark=CARD_DARK,
            button_secondary_background_fill_hover_dark=LINE_DARK,
            button_secondary_border_color=LINE,
            button_secondary_border_color_dark=LINE_DARK,
            button_secondary_text_color=INK,
            button_secondary_text_color_dark=INK_DARK,
            button_large_text_weight="500",
            checkbox_background_color_selected="*primary_600",
            checkbox_background_color_selected_dark="*primary_400",
            checkbox_border_color_selected="*primary_600",
            checkbox_border_color_selected_dark="*primary_400",
            checkbox_label_background_fill="transparent",
            checkbox_label_background_fill_dark="transparent",
            checkbox_label_border_width="0px",
            color_accent="*primary_600",
            color_accent_soft="*primary_50",
            color_accent_soft_dark="*primary_950",
            link_text_color="*primary_600",
            link_text_color_dark="*primary_300",
        )
