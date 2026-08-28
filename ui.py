import tkinter as tk
from tkinter import ttk

import pandas as pd

from model import RESULT_INFO


class CarPredictionUI:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        # =========================
        # Window Settings
        # =========================
        self.root.title("Car Evaluation - KNN Prediction")
        self.root.geometry("1450x900")
        self.root.minsize(1220, 720)
        self.root.configure(bg="#edf2fb")
        self.root.option_add("*Font", ("Arial", 10))

        # เปิดหน้าต่างแบบเต็มพื้นที่หน้าจอเท่าที่ทำได้
        # เพื่อให้มีพื้นที่แนวตั้งเพียงพอเสมอ ไม่ว่าจะเป็นจอ 1366x768 หรือ 1920x1080
        # (ผู้ใช้ยังสามารถย่อ/ปรับขนาดหน้าต่างเองได้ตามปกติ)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)
            except tk.TclError:
                pass

        self._configure_style()

        # ใช้ Grid กับ Root
        # row 0 = Header (คงที่)
        # row 1 = Main Area (ขยายได้ - กินพื้นที่ที่เหลือทั้งหมด)
        # row 2 = Nearest Neighbors (คงที่)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)

        self.root.grid_columnconfigure(0, weight=1)

        # =========================
        # Build UI
        # =========================
        self._build_header()
        self._build_main_layout()
        self._build_training_table()
        self._build_input_form()
        self._build_prediction_result()
        self._build_neighbor_table()

    # ==========================================================
    # STYLE
    # ==========================================================
    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Modern.Treeview",
            background="#ffffff",
            foreground="#1d3557",
            fieldbackground="#ffffff",
            rowheight=25,
            font=("Arial", 10),
        )

        style.map(
            "Modern.Treeview",
            background=[("selected", "#dfeeff")],
            foreground=[("selected", "#0b2545")],
        )

        style.configure(
            "Modern.Treeview.Heading",
            background="#e6eefc",
            foreground="#123456",
            font=("Arial", 10, "bold"),
        )

        style.map(
            "Modern.Treeview.Heading",
            background=[("active", "#dbe9ff")],
        )

        style.configure(
            "Custom.TCombobox",
            padding=4,
            fieldbackground="#ffffff",
            background="#ffffff",
        )

        style.map(
            "Custom.TCombobox",
            fieldbackground=[("readonly", "#ffffff")],
        )

    # ==========================================================
    # HEADER  (กระชับขึ้น เพื่อประหยัดพื้นที่แนวตั้งให้ Main Area)
    # ==========================================================
    def _build_header(self):

        self.header_frame = tk.Frame(
            self.root,
            bg="#123456",
            padx=22,
            pady=8,
        )

        self.header_frame.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        self.header_frame.grid_columnconfigure(0, weight=1)

        top_row = tk.Frame(self.header_frame, bg="#123456")
        top_row.grid(row=0, column=0, sticky="ew")
        top_row.grid_columnconfigure(1, weight=1)

        badge = tk.Label(
            top_row,
            text="KNN CLASSIFIER",
            bg="#1f6feb",
            fg="white",
            font=("Arial", 9, "bold"),
            padx=10,
            pady=3,
        )
        badge.grid(row=0, column=0, sticky="w")

        title_label = tk.Label(
            self.header_frame,
            text="CAR EVALUATION - KNN PREDICTION",
            bg="#123456",
            fg="white",
            font=("Arial", 21, "bold"),
        )

        title_label.grid(row=1, column=0, sticky="w", pady=(4, 2))

        summary = self.model.dataset_summary

        info_label = tk.Label(
            self.header_frame,
            text=(
                f"Dataset: {summary['total']:,} Records   |   "
                f"Training: {summary['training']:,}   |   "
                f"Testing: {summary['testing']:,}"
            ),
            bg="#123456",
            fg="#dfeaff",
            font=("Arial", 10),
        )

        info_label.grid(row=2, column=0, sticky="w")

        comparison_label = tk.Label(
            self.header_frame,
            text=(
                f"K = 1 : {self.model.accuracies[1] * 100:.2f}%    |    "
                f"K = 3 : {self.model.accuracies[3] * 100:.2f}%    |    "
                f"K = 5 : {self.model.accuracies[5] * 100:.2f}%"
            ),
            bg="#123456",
            fg="#fff0a7",
            font=("Arial", 10, "bold"),
        )

        comparison_label.grid(row=3, column=0, sticky="w", pady=(2, 0))

    # ==========================================================
    # MAIN LAYOUT
    # ==========================================================
    def _build_main_layout(self):

        self.main_frame = tk.Frame(
            self.root,
            bg="#edf2fb",
        )

        self.main_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=20,
            pady=10,
        )

        self.main_frame.grid_rowconfigure(0, weight=1)

        # ซ้าย = ตาราง Training Data
        self.main_frame.grid_columnconfigure(0, weight=3)

        # ขวา = Input + Prediction Result
        self.main_frame.grid_columnconfigure(1, weight=2)

    # ==========================================================
    # TRAINING DATA
    # ==========================================================
    def _build_training_table(self):

        self.left_frame = tk.LabelFrame(
            self.main_frame,
            text=" TRAINING DATA ",
            font=("Arial", 14, "bold"),
            bg="#ffffff",
            fg="#123456",
            padx=12,
            pady=8,
            bd=1,
            highlightbackground="#cfe0ff",
            highlightthickness=1,
        )

        self.left_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10),
        )

        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        tree_container = tk.Frame(
            self.left_frame,
            bg="#ffffff",
        )

        tree_container.grid(row=0, column=0, sticky="nsew")
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        self.data_columns = list(self.model.df.columns)

        self.tree = ttk.Treeview(
            tree_container,
            columns=self.data_columns,
            show="headings",
            style="Modern.Treeview",
        )

        for column in self.data_columns:

            self.tree.heading(
                column,
                text=column.upper(),
            )

            self.tree.column(
                column,
                width=105,
                anchor="center",
            )

        # แสดง Dataset ทั้งหมด
        for _, row in self.model.df.iterrows():

            self.tree.insert(
                "",
                "end",
                values=list(row),
            )

        # Vertical Scrollbar
        scrollbar_y = ttk.Scrollbar(
            tree_container,
            orient="vertical",
            command=self.tree.yview,
        )

        self.tree.configure(
            yscrollcommand=scrollbar_y.set,
        )

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")

    # ==========================================================
    # INPUT FORM
    # ==========================================================
    def _build_input_form(self):

        self.right_frame = tk.LabelFrame(
            self.main_frame,
            text=" INPUT CAR DATA ",
            font=("Arial", 14, "bold"),
            bg="#ffffff",
            fg="#123456",
            padx=16,
            pady=6,
            bd=1,
            highlightbackground="#cfe0ff",
            highlightthickness=1,
        )

        self.right_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(10, 0),
        )

        # Layout ภายใน right_frame:
        #   row 0-2 : Input fields (2 คอลัมน์ต่อแถว เพื่อประหยัดพื้นที่แนวตั้ง)
        #   row 3   : K Selection
        #   row 4   : Predict Button
        #   row 5   : Prediction Result   <-- ได้ weight=1 เพื่อรับพื้นที่ว่างทั้งหมด
        self.right_frame.grid_columnconfigure(0, weight=0)
        self.right_frame.grid_columnconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(2, weight=0)
        self.right_frame.grid_columnconfigure(3, weight=1)

        for r in range(5):
            self.right_frame.grid_rowconfigure(r, weight=0)

        # แถว Prediction Result: บังคับความสูงขั้นต่ำเสมอ (minsize)
        # เพื่อไม่ให้ถูกบีบจนมองไม่เห็น ไม่ว่าจอจะเล็กแค่ไหน
        self.right_frame.grid_rowconfigure(5, weight=1, minsize=190)

        # Variables
        self.buying_var = tk.StringVar(value="high")
        self.maint_var = tk.StringVar(value="high")
        self.doors_var = tk.StringVar(value="2")
        self.persons_var = tk.StringVar(value="2")
        self.lug_boot_var = tk.StringVar(value="small")
        self.safety_var = tk.StringVar(value="low")

        # K Default = 3
        self.k_var = tk.IntVar(value=3)

        # Dropdown - จัดเป็น 2 คอลัมน์ x 3 แถว เพื่อลดความสูงที่ใช้
        self._create_dropdown(
            "Buying Price:",
            self.buying_var,
            ["vhigh", "high", "med", "low"],
            row=0,
            col=0,
        )

        self._create_dropdown(
            "Maintenance:",
            self.maint_var,
            ["vhigh", "high", "med", "low"],
            row=0,
            col=2,
        )

        self._create_dropdown(
            "Doors:",
            self.doors_var,
            ["2", "3", "4", "5more"],
            row=1,
            col=0,
        )

        self._create_dropdown(
            "Persons:",
            self.persons_var,
            ["2", "4", "more"],
            row=1,
            col=2,
        )

        self._create_dropdown(
            "Luggage Boot:",
            self.lug_boot_var,
            ["small", "med", "big"],
            row=2,
            col=0,
        )

        self._create_dropdown(
            "Safety:",
            self.safety_var,
            ["low", "med", "high"],
            row=2,
            col=2,
        )

        # =========================
        # K Selection (รวมเป็นแถวเดียว)
        # =========================

        k_row = tk.Frame(self.right_frame, bg="#ffffff")

        k_row.grid(
            row=3,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(8, 4),
        )

        k_label = tk.Label(
            k_row,
            text="Number of Neighbors (K):",
            font=("Arial", 12, "bold"),
            fg="#123456",
            bg="#ffffff",
        )

        k_label.pack(side="left", padx=(0, 10))

        for k in [1, 3, 5]:

            radio = tk.Radiobutton(
                k_row,
                text=f"K = {k}",
                variable=self.k_var,
                value=k,
                font=("Arial", 10),
                bg="#ffffff",
                fg="#123456",
                selectcolor="#dfeeff",
                activebackground="#ffffff",
            )

            radio.pack(
                side="left",
                padx=8,
            )

        # =========================
        # Predict Button
        # =========================

        self.predict_button = tk.Button(
            self.right_frame,
            text="PREDICT",
            font=("Arial", 15, "bold"),
            cursor="hand2",
            command=self.predict_car,
            bg="#1f6feb",
            fg="white",
            activebackground="#1859c7",
            activeforeground="white",
            bd=0,
            padx=30,
            pady=7,
            relief="flat",
        )

        self.predict_button.grid(
            row=4,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(4, 6),
        )

    # ==========================================================
    # CREATE DROPDOWN (label ซ้าย + combobox ขวา ในกลุ่มคอลัมน์ที่กำหนด)
    # ==========================================================
    def _create_dropdown(
        self,
        label_text,
        variable,
        values,
        row,
        col,
    ):

        label = tk.Label(
            self.right_frame,
            text=label_text,
            font=("Arial", 10, "bold"),
            fg="#123456",
            bg="#ffffff",
        )

        label.grid(
            row=row,
            column=col,
            sticky="w",
            padx=(0 if col == 0 else 18, 6),
            pady=3,
        )

        dropdown = ttk.Combobox(
            self.right_frame,
            textvariable=variable,
            values=values,
            state="readonly",
            width=10,
            style="Custom.TCombobox",
        )

        dropdown.grid(
            row=row,
            column=col + 1,
            sticky="ew",
            pady=3,
        )

    # ==========================================================
    # PREDICTION RESULT
    # ==========================================================
    def _build_prediction_result(self):

        self.result_panel = tk.Frame(
            self.right_frame,
            bg="#eef6ff",
            bd=1,
            highlightbackground="#cfe0ff",
            highlightthickness=1,
        )

        self.result_panel.grid(
            row=5,
            column=0,
            columnspan=4,
            sticky="nsew",
            pady=(4, 0),
        )

        # เนื้อหาถูกจัดกลางในแนวตั้งด้วย pack(expand=True)
        # เพื่อให้ดูดีทั้งตอนพื้นที่พอดี และตอนพื้นที่เหลือเยอะ (จอใหญ่)
        # แต่จะไม่ถูกตัดเพราะ result_panel มีขนาดขั้นต่ำตามเนื้อหาเสมอ
        content = tk.Frame(self.result_panel, bg="#eef6ff")
        content.pack(expand=True, fill="both", padx=15, pady=10)

        # Title
        self.result_title = tk.Label(
            content,
            text="PREDICTION RESULT",
            font=("Arial", 12, "bold"),
            bg="#eef6ff",
            fg="#123456",
        )

        self.result_title.pack()

        # Main Result
        self.result_label = tk.Label(
            content,
            text="READY",
            font=("Arial", 21, "bold"),
            bg="#eef6ff",
            fg="#0f4d8f",
        )

        self.result_label.pack(
            pady=(4, 2),
        )

        # Description (รองรับหลายบรรทัด ไม่ถูกตัด)
        self.result_detail_label = tk.Label(
            content,
            text="เลือกข้อมูลรถ แล้วกด Predict",
            font=("Arial", 10),
            bg="#eef6ff",
            fg="#314e6d",
            justify="center",
            wraplength=380,
        )

        self.result_detail_label.pack(
            pady=(0, 6),
            fill="x",
        )

        # Accuracy
        self.accuracy_label = tk.Label(
            content,
            text="",
            font=("Arial", 10, "bold"),
            bg="#eef6ff",
            fg="#1d3557",
            wraplength=380,
            justify="center",
        )

        self.accuracy_label.pack(
            pady=(0, 4),
        )

        # Vote Result
        self.vote_label = tk.Label(
            content,
            text="",
            font=("Arial", 10),
            justify="center",
            bg="#eef6ff",
            fg="#0d3b66",
            wraplength=380,
        )

        self.vote_label.pack(
            pady=(0, 2),
        )

    # ==========================================================
    # NEAREST NEIGHBORS TABLE
    # ==========================================================
    def _build_neighbor_table(self):

        self.neighbor_frame = tk.LabelFrame(
            self.root,
            text=" NEAREST NEIGHBORS USED FOR PREDICTION ",
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            fg="#123456",
            padx=8,
            pady=4,
            bd=1,
            highlightbackground="#cfe0ff",
            highlightthickness=1,
        )

        # ตารางนี้จงใจทำให้ "เตี้ยและกระชับ" (ไม่กิน weight ใดๆ จาก root)
        # เพื่อคืนพื้นที่แนวตั้งด้านบนให้ Prediction Result เป็นหลัก
        # ถ้า K = 5 แล้วแถวไม่พอดี จะมี scrollbar แนวตั้งให้เลื่อนดูแทน
        self.neighbor_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 8),
        )

        self.neighbor_frame.grid_columnconfigure(0, weight=1)
        self.neighbor_frame.grid_rowconfigure(0, weight=0)

        self.neighbor_columns = [
            "number",
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
            "class",
            "distance",
        ]

        # height=3 เพื่อให้ตารางนี้กระชับ ไม่แย่งพื้นที่แนวตั้งจาก Prediction Result
        # ถ้าเลือก K = 5 จะมี scrollbar แนวตั้งให้เลื่อนดูแถวที่เหลือ
        self.neighbor_tree = ttk.Treeview(
            self.neighbor_frame,
            columns=self.neighbor_columns,
            show="headings",
            height=3,
            style="Modern.Treeview",
        )

        neighbor_headings = [
            "#",
            "BUYING",
            "MAINT",
            "DOORS",
            "PERSONS",
            "LUG_BOOT",
            "SAFETY",
            "CLASS",
            "DISTANCE",
        ]

        for column, heading in zip(
            self.neighbor_columns,
            neighbor_headings,
        ):

            self.neighbor_tree.heading(
                column,
                text=heading,
            )

            width = 100 if column == "distance" else 90

            self.neighbor_tree.column(
                column,
                width=width,
                minwidth=70,
                anchor="center",
            )

        x_scrollbar = ttk.Scrollbar(
            self.neighbor_frame,
            orient="horizontal",
            command=self.neighbor_tree.xview,
        )

        y_scrollbar = ttk.Scrollbar(
            self.neighbor_frame,
            orient="vertical",
            command=self.neighbor_tree.yview,
        )

        self.neighbor_tree.configure(
            xscrollcommand=x_scrollbar.set,
            yscrollcommand=y_scrollbar.set,
        )

        self.neighbor_tree.grid(row=0, column=0, sticky="ew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")

    # ==========================================================
    # PREDICT
    # ==========================================================
    def predict_car(self):

        # K ที่ผู้ใช้เลือก
        selected_k = self.k_var.get()

        # Model ที่ตรงกับ K
        model = self.model.models[selected_k]

        # รับค่าจาก GUI
        new_car = pd.DataFrame([
            {
                "buying": self.buying_var.get(),
                "maint": self.maint_var.get(),
                "doors": self.doors_var.get(),
                "persons": self.persons_var.get(),
                "lug_boot": self.lug_boot_var.get(),
                "safety": self.safety_var.get(),
            }
        ])

        # Prediction
        prediction, vote_text, _ = self.model.predict(
            selected_k,
            new_car,
        )

        # =========================
        # Update Result
        # =========================

        self.result_label.config(
            text=RESULT_INFO[prediction]["title"],
        )

        self.result_detail_label.config(
            text=RESULT_INFO[prediction]["detail"],
        )

        self.accuracy_label.config(
            text=(
                f"Selected Model: K = {selected_k}"
                f"   |   "
                f"Accuracy: "
                f"{self.model.accuracies[selected_k] * 100:.2f}%"
            )
        )

        self.vote_label.config(
            text=vote_text,
        )

        # =========================
        # Find Nearest Neighbors
        # =========================

        encoded_car = self.model.encoder.transform(
            new_car
        )

        distances, indices = model.kneighbors(
            encoded_car
        )

        # ล้างข้อมูลเก่า
        for item in self.neighbor_tree.get_children():
            self.neighbor_tree.delete(item)

        # เพิ่มข้อมูลเพื่อนบ้านใหม่
        for number, index in enumerate(
            indices[0],
            start=1,
        ):

            neighbor_data = self.model.X_train.iloc[index]

            neighbor_class = self.model.y_train.iloc[index]

            distance = distances[0][number - 1]

            self.neighbor_tree.insert(
                "",
                "end",
                values=[
                    number,
                    neighbor_data["buying"],
                    neighbor_data["maint"],
                    neighbor_data["doors"],
                    neighbor_data["persons"],
                    neighbor_data["lug_boot"],
                    neighbor_data["safety"],
                    neighbor_class,
                    f"{distance:.2f}",
                ],
            )
