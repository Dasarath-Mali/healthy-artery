"""
=============================================================================
Healthy Artery - Desktop GUI Application
=============================================================================
A modern, intuitive desktop GUI for cardiovascular disease risk prediction
and model benchmarking using CustomTkinter.

Key Features:
- Pre-populated realistic patient defaults on startup with instant prediction.
- Auto-reprediction on model change and Enter key press.
- Clear, real-time input validation with red borders and inline warnings.
- Prominent risk tier badges: Green (<30%), Amber (30%-60%), Red (>60%).
- Model Comparison tab with live metrics table and high-res chart browser.
- Light and Dark mode support.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image
import customtkinter as ctk

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import (
    load_trained_models,
    validate_input_values,
    predict_single_instance,
    FEATURE_DEFINITIONS,
    MODEL_REGISTRY
)
from src.config import NUMERIC_FEATURES

# Set initial appearance and theme
ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


class HealthyArteryApp(ctk.CTk):
    """
    Main application window for the Healthy Artery Risk Predictor.
    """

    def __init__(self):
        super().__init__()

        # ---------------------------------------------------------------------
        # 1. Window Configuration
        # ---------------------------------------------------------------------
        self.title("Heart disease risk predictor")
        self.geometry("1060x800")
        self.minsize(940, 700)

        # ---------------------------------------------------------------------
        # 2. Model Loading (Executed Once at Startup)
        # ---------------------------------------------------------------------
        try:
            self.models = load_trained_models()
        except Exception as e:
            self.withdraw()
            messagebox.showerror(
                "Model Loading Error",
                f"Failed to load trained models:\n\n{e}\n\n"
                "Please make sure you have run 'python -m src.train' first."
            )
            self.destroy()
            sys.exit(1)

        # Widget references
        self.input_entries = {}
        self.error_labels = {}
        self.current_chart_image = None

        # ---------------------------------------------------------------------
        # 3. Top Header Bar (Branding & Theme Switcher)
        # ---------------------------------------------------------------------
        self._build_header()

        # ---------------------------------------------------------------------
        # 4. Main Tabview (Predictor & Model Comparison)
        # ---------------------------------------------------------------------
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # Create the two tabs
        self.tab_predictor = self.tabview.add("Risk Assessment")
        self.tab_comparison = self.tabview.add("Model Comparison")

        # Build each tab's contents
        self._build_predictor_tab()
        self._build_comparison_tab()

        # Pre-fill sample values and run initial prediction so user sees instant output
        self._on_load_sample_clicked()

    # =========================================================================
    # Header Construction
    # =========================================================================
    def _build_header(self):
        """Build the top header with tool branding and appearance switcher."""
        header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header_frame.pack(fill="x", padx=25, pady=(15, 8))

        # Title & Subtitle
        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.pack(side="left")

        app_title = ctk.CTkLabel(
            title_box,
            text="Healthy Artery",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        app_title.pack(anchor="w")

        app_subtitle = ctk.CTkLabel(
            title_box,
            text="Cardiovascular Disease Risk Prediction & Model Analysis System",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        app_subtitle.pack(anchor="w")

        # Theme Switcher (Light / Dark)
        theme_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        theme_box.pack(side="right", pady=5)

        theme_label = ctk.CTkLabel(theme_box, text="Appearance:", font=ctk.CTkFont(size=12))
        theme_label.pack(side="left", padx=(0, 8))

        self.theme_option = ctk.CTkOptionMenu(
            theme_box,
            values=["System", "Light", "Dark"],
            command=self._change_appearance_mode,
            width=100,
            height=28
        )
        self.theme_option.set("System")
        self.theme_option.pack(side="left")

    def _change_appearance_mode(self, mode_name):
        """Callback to switch between Light and Dark themes."""
        ctk.set_appearance_mode(mode_name)

    # =========================================================================
    # Tab 1: Predictor (Risk Assessment)
    # =========================================================================
    def _build_predictor_tab(self):
        """Construct the prediction form, controls, and risk result display."""
        container = ctk.CTkFrame(self.tab_predictor, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)
        container.grid_columnconfigure(0, weight=5)
        container.grid_columnconfigure(1, weight=5)
        container.grid_rowconfigure(0, weight=1)

        # ---------------------------------------------------------------------
        # Left Column: Input Form & Model Selector
        # ---------------------------------------------------------------------
        left_card = ctk.CTkFrame(container, corner_radius=12)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        # Form Title
        form_title = ctk.CTkLabel(
            left_card,
            text="Clinical Input Parameters",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        form_title.pack(anchor="w", padx=20, pady=(15, 10))

        # Model Selection Dropdown
        model_label = ctk.CTkLabel(
            left_card,
            text="Select Trained Classifier:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        model_label.pack(anchor="w", padx=20, pady=(0, 2))

        self.model_dropdown = ctk.CTkOptionMenu(
            left_card,
            values=list(self.models.keys()),
            command=self._on_model_changed,
            width=340,
            height=32
        )
        self.model_dropdown.set("XGBoost")
        self.model_dropdown.pack(anchor="w", padx=20, pady=(0, 15))

        # Inputs container (direct frame for immediate visibility without inner scrolling)
        inputs_frame = ctk.CTkFrame(left_card, fg_color="transparent")
        inputs_frame.pack(fill="both", expand=True, padx=20, pady=0)

        for feat_name in NUMERIC_FEATURES:
            meta = FEATURE_DEFINITIONS[feat_name]

            # Row header: name + range description
            row_header = ctk.CTkFrame(inputs_frame, fg_color="transparent")
            row_header.pack(fill="x", pady=(6, 1))

            lbl = ctk.CTkLabel(
                row_header,
                text=f"{meta['display_name']} ({meta['unit']})",
                font=ctk.CTkFont(size=13, weight="bold")
            )
            lbl.pack(side="left")

            range_hint = ctk.CTkLabel(
                row_header,
                text=f"[{meta['min']} - {meta['max']} {meta['unit']}]",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            range_hint.pack(side="right")

            # Entry box
            entry = ctk.CTkEntry(
                inputs_frame,
                placeholder_text=f"e.g. {meta['default']}",
                width=340,
                height=32
            )
            entry.pack(anchor="w", pady=(0, 2))
            # Bind Enter key so hitting Enter triggers prediction immediately
            entry.bind("<Return>", lambda event: self._on_predict_clicked())
            self.input_entries[feat_name] = entry

            # Inline error text
            err_label = ctk.CTkLabel(
                inputs_frame,
                text="",
                font=ctk.CTkFont(size=11),
                text_color="#e74c3c"
            )
            err_label.pack(anchor="w", pady=(0, 2))
            self.error_labels[feat_name] = err_label

        # Action Buttons (Predict, Clear, Load Sample)
        btn_frame = ctk.CTkFrame(left_card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 15))

        self.predict_btn = ctk.CTkButton(
            btn_frame,
            text="Predict Risk",
            command=self._on_predict_clicked,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2980b9",
            hover_color="#1f618d",
            height=38,
            width=130
        )
        self.predict_btn.pack(side="left", padx=(0, 8))

        self.clear_btn = ctk.CTkButton(
            btn_frame,
            text="Clear",
            command=self._on_clear_clicked,
            font=ctk.CTkFont(size=13),
            fg_color="#7f8c8d",
            hover_color="#616a6b",
            height=38,
            width=80
        )
        self.clear_btn.pack(side="left", padx=(0, 8))

        self.sample_btn = ctk.CTkButton(
            btn_frame,
            text="Sample Values",
            command=self._on_load_sample_clicked,
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
            border_width=1,
            height=38,
            width=120
        )
        self.sample_btn.pack(side="left")

        # ---------------------------------------------------------------------
        # Right Column: Prediction Result Area
        # ---------------------------------------------------------------------
        right_card = ctk.CTkFrame(container, corner_radius=12)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)

        result_heading = ctk.CTkLabel(
            right_card,
            text="Assessment Result",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        result_heading.pack(anchor="w", padx=25, pady=(15, 10))

        # Dynamic Result Box
        self.result_box = ctk.CTkFrame(right_card, corner_radius=10, fg_color=("gray90", "gray17"))
        self.result_box.pack(fill="x", padx=25, pady=10)

        self.result_model_label = ctk.CTkLabel(
            self.result_box,
            text="Evaluated with: XGBoost",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.result_model_label.pack(pady=(18, 5))

        self.risk_pct_label = ctk.CTkLabel(
            self.result_box,
            text="-- %",
            font=ctk.CTkFont(size=46, weight="bold")
        )
        self.risk_pct_label.pack(pady=5)

        # Colored Risk Label Badge
        self.risk_badge = ctk.CTkLabel(
            self.result_box,
            text="NO PREDICTION",
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=8,
            fg_color="gray50",
            text_color="white",
            padx=18,
            pady=7
        )
        self.risk_badge.pack(pady=(5, 18))

        # Legend explanation for colors
        legend_frame = ctk.CTkFrame(right_card, fg_color="transparent")
        legend_frame.pack(fill="x", padx=25, pady=10)

        legend_title = ctk.CTkLabel(
            legend_frame,
            text="Risk Classification Scale:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        legend_title.pack(anchor="w", pady=(0, 6))

        l_green = ctk.CTkLabel(
            legend_frame,
            text="● Green: Low Risk (< 30% probability)",
            font=ctk.CTkFont(size=12),
            text_color="#2ecc71"
        )
        l_green.pack(anchor="w")

        l_amber = ctk.CTkLabel(
            legend_frame,
            text="● Amber: Moderate Risk (30% - 60% probability)",
            font=ctk.CTkFont(size=12),
            text_color="#f39c12"
        )
        l_amber.pack(anchor="w")

        l_red = ctk.CTkLabel(
            legend_frame,
            text="● Red: High Risk (> 60% probability)",
            font=ctk.CTkFont(size=12),
            text_color="#e74c3c"
        )
        l_red.pack(anchor="w")

        # Input Parameter Echo Card
        self.echo_frame = ctk.CTkFrame(right_card, corner_radius=10, fg_color="transparent")
        self.echo_frame.pack(fill="both", expand=True, padx=25, pady=10)

        self.echo_label = ctk.CTkLabel(
            self.echo_frame,
            text="",
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        self.echo_label.pack(anchor="w")

        # Mandatory Regulatory Disclaimer
        disclaimer_box = ctk.CTkFrame(right_card, corner_radius=8, fg_color=("gray85", "gray20"))
        disclaimer_box.pack(fill="x", side="bottom", padx=25, pady=18)

        disclaimer_text = ctk.CTkLabel(
            disclaimer_box,
            text="This is an educational tool, not a medical diagnosis.",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#c0392b", "#e74c3c"),
            pady=10
        )
        disclaimer_text.pack()

    # =========================================================================
    # Prediction Event Handlers
    # =========================================================================
    def _on_model_changed(self, choice):
        """Automatically recalculate risk when user chooses another model."""
        self._on_predict_clicked()

    def _on_predict_clicked(self):
        """Handle Predict button click: validate inputs and calculate risk."""
        # 1. Reset previous error states
        for feat in NUMERIC_FEATURES:
            self.error_labels[feat].configure(text="")
            # Reset entry border color
            self.input_entries[feat].configure(border_color=["#979DA2", "#565B5E"])

        # 2. Gather raw inputs
        raw_inputs = {feat: self.input_entries[feat].get() for feat in NUMERIC_FEATURES}

        # 3. Validate
        is_valid, cleaned_values, errors = validate_input_values(raw_inputs)

        if not is_valid:
            # Highlight fields with invalid entries
            for feat, err_msg in errors.items():
                self.error_labels[feat].configure(text=f"⚠ {err_msg}")
                self.input_entries[feat].configure(border_color="#e74c3c")

            # Update result box to clearly show validation issues
            self.result_model_label.configure(
                text="⚠ Validation Alert: Some fields need attention",
                text_color="#e74c3c"
            )
            self.risk_pct_label.configure(text="Check Inputs", text_color="#e74c3c")
            self.risk_badge.configure(text="INVALID INPUT", fg_color="#e74c3c")

            err_summary = "\n".join(f" • {FEATURE_DEFINITIONS[f]['display_name']}: {msg}" for f, msg in errors.items())
            self.echo_label.configure(
                text=f"Please correct the following fields:\n{err_summary}",
                text_color="#e74c3c"
            )
            return

        # 4. Predict with selected model
        selected_model_name = self.model_dropdown.get()
        model_pipeline = self.models[selected_model_name]

        try:
            result = predict_single_instance(model_pipeline, cleaned_values)
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n\n{e}")
            return

        # 5. Update GUI with prediction outputs
        pct = result["risk_percentage"]
        category = result["risk_category"]
        color = result["status_color"]

        self.result_model_label.configure(
            text=f"Evaluated with: {selected_model_name}",
            text_color="gray"
        )
        self.risk_pct_label.configure(text=f"{pct:.1f} %", text_color=color)

        badge_text = f"{category.upper()} ({pct:.1f}%)"
        self.risk_badge.configure(text=badge_text, fg_color=color)

        # Echo input summary
        echo_lines = [
            f"Patient Age: {cleaned_values['age']:.0f} yrs",
            f"Resting BP: {cleaned_values['trestbps']:.0f} mm Hg",
            f"Serum Cholesterol: {cleaned_values['chol']:.0f} mg/dl",
            f"Max Heart Rate: {cleaned_values['thalach']:.0f} bpm",
            f"ST Depression: {cleaned_values['oldpeak']:.1f} mm"
        ]
        self.echo_label.configure(
            text="Evaluated Profile:\n" + "\n".join(f" • {l}" for l in echo_lines),
            text_color=("gray10", "gray90")
        )

    def _on_clear_clicked(self):
        """Reset all input fields, errors, and output labels."""
        for feat in NUMERIC_FEATURES:
            self.input_entries[feat].delete(0, tk.END)
            self.error_labels[feat].configure(text="")
            self.input_entries[feat].configure(border_color=["#979DA2", "#565B5E"])

        self.result_model_label.configure(text="Awaiting user input...", text_color="gray")
        self.risk_pct_label.configure(text="-- %", text_color=("gray10", "gray90"))
        self.risk_badge.configure(text="NO PREDICTION", fg_color="gray50")
        self.echo_label.configure(text="")

    def _on_load_sample_clicked(self):
        """Populate input fields with standard realistic sample values and predict immediately."""
        self._on_clear_clicked()
        sample_values = {
            "age": "55",
            "trestbps": "140",
            "chol": "250",
            "thalach": "150",
            "oldpeak": "1.5"
        }
        for feat, val in sample_values.items():
            self.input_entries[feat].insert(0, val)

        # Automatically execute prediction so output is immediately visible
        self._on_predict_clicked()

    # =========================================================================
    # Tab 2: Model Comparison (Metrics Table & Saved Visualizations)
    # =========================================================================
    def _build_comparison_tab(self):
        """Construct the Model Comparison tab with metrics table and saved figures."""
        scroll_container = ctk.CTkScrollableFrame(self.tab_comparison, corner_radius=10)
        scroll_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Section Heading
        table_title = ctk.CTkLabel(
            scroll_container,
            text="Model Performance Benchmarks",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        table_title.pack(anchor="w", padx=15, pady=(10, 5))

        table_subtitle = ctk.CTkLabel(
            scroll_container,
            text="Held-out 20% test set (N=61) and 5x5 Repeated Stratified Cross-Validation (N=303):",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        table_subtitle.pack(anchor="w", padx=15, pady=(0, 10))

        # Metrics Table (Structured Grid)
        table_card = ctk.CTkFrame(scroll_container, corner_radius=10)
        table_card.pack(fill="x", padx=15, pady=(0, 15))

        headers = [
            "Model Name", "ROC-AUC", "Recall", "Precision", "F1 Score",
            "Accuracy", "Brier", "FN", "FP", "5x5 Repeated CV ROC-AUC"
        ]

        # Table Header
        for col_idx, h_text in enumerate(headers):
            lbl = ctk.CTkLabel(
                table_card,
                text=h_text,
                font=ctk.CTkFont(size=12, weight="bold"),
                padx=8,
                pady=8
            )
            lbl.grid(row=0, column=col_idx, sticky="nsew", padx=2, pady=2)
            table_card.grid_columnconfigure(col_idx, weight=1)

        metrics_rows = [
            ["Logistic Regression", "0.873", "78.6%", "81.5%", "0.800", "82.0%", "0.185", "6", "5", "0.788 ± 0.037"],
            ["Random Forest", "0.897", "82.1%", "88.5%", "0.852", "86.9%", "0.146", "5", "3", "0.786 ± 0.039"],
            ["XGBoost", "0.902", "82.1%", "88.5%", "0.852", "86.9%", "0.141", "5", "3", "0.784 ± 0.043"]
        ]

        for r_idx, row in enumerate(metrics_rows, start=1):
            for c_idx, val in enumerate(row):
                is_bold = (c_idx == 0)
                cell = ctk.CTkLabel(
                    table_card,
                    text=val,
                    font=ctk.CTkFont(size=12, weight="bold" if is_bold else "normal"),
                    padx=6,
                    pady=6
                )
                cell.grid(row=r_idx, column=c_idx, sticky="nsew", padx=2, pady=1)

        # ---------------------------------------------------------------------
        # Visualizations Viewer
        # ---------------------------------------------------------------------
        charts_title = ctk.CTkLabel(
            scroll_container,
            text="Saved Evaluation Charts & Curves",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        charts_title.pack(anchor="w", padx=15, pady=(15, 5))

        chart_ctrl_frame = ctk.CTkFrame(scroll_container, fg_color="transparent")
        chart_ctrl_frame.pack(fill="x", padx=15, pady=(0, 10))

        sel_label = ctk.CTkLabel(chart_ctrl_frame, text="Select Chart to View:", font=ctk.CTkFont(size=13))
        sel_label.pack(side="left", padx=(0, 10))

        self.chart_files = {
            "ROC & Precision-Recall Curves": "reports/figures/roc_pr_curves.png",
            "Confusion Matrices": "reports/figures/confusion_matrices.png",
            "Threshold Sweep (FN vs FP)": "reports/figures/threshold_sweep.png",
            "SHAP Feature Importance (XGBoost)": "reports/figures/shap_summary_xgb.png",
            "Calibration Curves": "reports/figures/calibration_curves.png",
            "Core vs Extension Feature Comparison": "reports/figures/core_vs_extension.png",
            "Feature Distributions by Disease Status": "reports/figures/numeric_distributions.png",
            "Correlation Heatmap": "reports/figures/correlation_heatmap.png"
        }

        self.chart_dropdown = ctk.CTkOptionMenu(
            chart_ctrl_frame,
            values=list(self.chart_files.keys()),
            command=self._load_selected_chart,
            width=320,
            height=30
        )
        self.chart_dropdown.set("ROC & Precision-Recall Curves")
        self.chart_dropdown.pack(side="left")

        # Image Display Container
        self.image_container = ctk.CTkFrame(scroll_container, corner_radius=10)
        self.image_container.pack(fill="both", expand=True, padx=15, pady=10)

        self.image_label = ctk.CTkLabel(self.image_container, text="")
        self.image_label.pack(padx=10, pady=10)

        # Load initial chart
        self._load_selected_chart("ROC & Precision-Recall Curves")

    def _load_selected_chart(self, chart_name):
        """Load and display the selected PNG chart from reports/figures/."""
        rel_path = self.chart_files.get(chart_name)
        if not rel_path or not os.path.exists(rel_path):
            self.image_label.configure(
                text=f"Chart file not found: {rel_path}\nPlease run the analysis scripts.",
                image=""
            )
            return

        try:
            pil_img = Image.open(rel_path)
            orig_w, orig_h = pil_img.size

            max_w, max_h = 820, 480
            scale = min(max_w / orig_w, max_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            ctk_img = ctk.CTkImage(
                light_image=pil_img,
                dark_image=pil_img,
                size=(new_w, new_h)
            )

            self.current_chart_image = ctk_img
            self.image_label.configure(image=ctk_img, text="")
        except Exception as e:
            self.image_label.configure(
                text=f"Error displaying image:\n{e}",
                image=""
            )


def main():
    """Start the Healthy Artery CustomTkinter application."""
    app = HealthyArteryApp()
    app.mainloop()


if __name__ == "__main__":
    main()
