import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StudentGradeWeightCalculator:
    def __init__(self, root):
        self.root = root
        self.setup_frame = None
        self.root.title("Student Grades Weight Calculator")
        
        # Center window
        width, height = 800, 600
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x, y = (sw - width)//2, (sh - height)//2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.configure(padx=20, pady=20)

        # Data storage
        self.component_names = []
        self.data_matrix = []
        self.final_grades = []
        self.weights = None
        self.num_components = 0

        # Notebook & tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # For each tab: make it scrollable
        self.setup_tab = self._make_scrollable_tab("Setup")
        self.data_tab  = self._make_scrollable_tab("Enter Data")
        self.results_tab = self._make_scrollable_tab("Results")

        
        # Initialize each tab
        self.setup_setup_tab()
        self.setup_data_tab()
        self.setup_results_tab()
        
    def _make_scrollable_tab(self, title):
        """Helper to create a new notebook tab that's vertically scrollable,
        and also makes its content frame always match the canvas width."""
        container = ttk.Frame(self.notebook)
        self.notebook.add(container, text=title)

        # 1. Canvas and scrollbar
        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0, takefocus=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # 2. Frame inside the canvas
        content = ttk.Frame(canvas)
        # save the window ID so we can resize it later
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        # 3. Whenever content size changes, update scrollregion
        def on_content_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        content.bind("<Configure>", on_content_configure)

        # 4. Whenever canvas width changes, resize the inner window
        def on_canvas_configure(event):
            # event.width is the new width of the canvas
            canvas.itemconfigure(window_id, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        # Store references
        container.canvas = canvas
        container.content = content
        return container


    def setup_setup_tab(self):
        parent = self.setup_tab.content

        # 1. Clear everything
        for w in parent.winfo_children():
            w.destroy()

        # 2. Main setup frame
        self.setup_frame = ttk.LabelFrame(parent, text="Grade Components Setup")
        self.setup_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 3. Component‐count prompt
        ttk.Label(self.setup_frame,
                text="Enter the number of grade components (e.g., homework, quizzes, exams):"
        ).pack(pady=10)
        self.num_components_var = tk.StringVar()
        entry = tk.Entry(self.setup_frame,
                        textvariable=self.num_components_var,
                        width=5,
                        bd=1,               # normal border
                        highlightthickness=0  # NO focus ring
        )
        entry.pack(pady=5)
        ttk.Button(self.setup_frame,
                text="Set Components",
                command=self.set_components
        ).pack(pady=10)

        # 4. Placeholder for dynamic entries
        self.components_frame = ttk.Frame(self.setup_frame)
        self.components_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 5. Instructions – make this fill *all* horizontal space
        instr = ttk.LabelFrame(parent, text="Instructions")
        instr.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)       # ← expand here!

        instruction_text = (
            "This application helps determine the weights of different grading components.\n\n"
            "How to use:\n"
            "1. Enter the number of grading components (homework, quizzes, etc.).\n"
            "2. Name each component.\n"
            "3. Enter student scores for each component and their final grades.\n"
            "4. Calculate to find the weight of each component.\n\n"
            "The application uses a linear system (Ax = b) where:\n"
            "- A is the matrix of student scores\n"
            "- x is the vector of weights we want to find\n"
            "- b is the vector of final grades\n\n"
            "The system finds weights that best fit the provided data."
        )

        # Remove fixed wraplength, and let the label fill its parent
        lbl = ttk.Label(instr, text=instruction_text, justify=tk.LEFT)
        lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


    def set_components(self):
        try:
            self.num_components = int(self.num_components_var.get())
            if self.num_components < 1:
                messagebox.showerror("Error", "Number of components must be at least 1")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            return
        
        # Clear the existing components frame
        for widget in self.components_frame.winfo_children():
            widget.destroy()
        
        # Create entries for component names
        self.component_entries = []
        for i in range(self.num_components):
            frame = ttk.Frame(self.components_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"Component {i+1} name:").pack(side=tk.LEFT, padx=5)
            entry = ttk.Entry(frame, width=20)
            entry.pack(side=tk.LEFT, padx=5)
            self.component_entries.append(entry)
        
        # Add button to confirm component names
        ttk.Button(self.components_frame, text="Confirm Components", 
                   command=self.confirm_components).pack(pady=10, anchor='center')
        
        ttk.Button(self.components_frame, text="Back", 
                   command=self.setup_setup_tab).pack(pady=10, anchor='center')

    def confirm_components(self):
        # Collect component names
        self.component_names = []
        for entry in self.component_entries:
            name = entry.get().strip()
            if not name:
                messagebox.showerror("Error", "All component names must be filled")
                return
            self.component_names.append(name)
        
        # Set up the data entry tab
        self.setup_data_entry()
        
        # Switch to the data tab
        self.notebook.select(self.data_tab)

    def setup_data_tab(self):
        self.data_frame = ttk.Frame(self.data_tab)
        self.data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # This will be populated after components are set

    def setup_data_entry(self):
        # Clear the existing data frame
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        # Create a frame for the data table
        table_frame = ttk.Frame(self.data_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollbar and canvas for the table
        canvas = tk.Canvas(table_frame)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create headers
        ttk.Label(scrollable_frame, text="Student", width=10).grid(row=0, column=0, padx=5, pady=5)
        for i, name in enumerate(self.component_names):
            ttk.Label(scrollable_frame, text=name, width=10).grid(row=0, column=i+1, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Final Grade", width=10).grid(row=0, column=len(self.component_names)+1, padx=5, pady=5)
        
        # Create entries for initial students (let's start with 5)
        self.student_entries = []
        num_initial_students = 5
        
        for i in range(num_initial_students):
            row_entries = []
            ttk.Label(scrollable_frame, text=f"Student {i+1}", width=10).grid(row=i+1, column=0, padx=5, pady=2)
            
            # Component scores
            for j in range(self.num_components):
                entry = ttk.Entry(scrollable_frame, width=10)
                entry.grid(row=i+1, column=j+1, padx=5, pady=2)
                row_entries.append(entry)
            
            # Final grade
            final_entry = ttk.Entry(scrollable_frame, width=10)
            final_entry.grid(row=i+1, column=self.num_components+1, padx=5, pady=2)
            row_entries.append(final_entry)
            
            self.student_entries.append(row_entries)
        
        # Add buttons for data management
        button_frame = ttk.Frame(self.data_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Add Student", command=self.add_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove Last Student", command=self.remove_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All Data", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Calculate Weights", command=self.calculate_weights).pack(side=tk.RIGHT, padx=5)
        
        # Store the scrollable frame and number of students for later use
        self.scrollable_frame = scrollable_frame
        self.num_students = num_initial_students

    def add_student(self):
        i = self.num_students
        row_entries = []
        
        ttk.Label(self.scrollable_frame, text=f"Student {i+1}", width=10).grid(row=i+1, column=0, padx=5, pady=2)
        
        # Component scores
        for j in range(self.num_components):
            entry = ttk.Entry(self.scrollable_frame, width=10)
            entry.grid(row=i+1, column=j+1, padx=5, pady=2)
            row_entries.append(entry)
        
        # Final grade
        final_entry = ttk.Entry(self.scrollable_frame, width=10)
        final_entry.grid(row=i+1, column=self.num_components+1, padx=5, pady=2)
        row_entries.append(final_entry)
        
        self.student_entries.append(row_entries)
        self.num_students += 1

    def remove_student(self):
        if self.num_students > 1:
            # Remove entries from the grid
            for entry in self.student_entries[-1]:
                entry.destroy()
            
            # Remove the student label
            for widget in self.scrollable_frame.grid_slaves():
                if int(widget.grid_info()["row"]) == self.num_students:
                    if int(widget.grid_info()["column"]) == 0:
                        widget.destroy()
            
            # Update the list and counter
            self.student_entries.pop()
            self.num_students -= 1

    def clear_data(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all data?"):
            for row in self.student_entries:
                for entry in row:
                    entry.delete(0, tk.END)

    def collect_data(self):
        data_matrix = []
        final_grades = []
        
        for row in self.student_entries:
            component_scores = []
            try:
                # Collect component scores
                for i in range(self.num_components):
                    value = float(row[i].get())
                    component_scores.append(value)
                
                # Collect final grade
                final_grade = float(row[-1].get())
                
                # Add to matrices
                data_matrix.append(component_scores)
                final_grades.append(final_grade)
            except ValueError:
                # Skip rows with incomplete or invalid data
                continue
        
        return np.array(data_matrix), np.array(final_grades)

    def calculate_weights(self):
        # Collect data
        A, b = self.collect_data()
        
        if len(A) < self.num_components:
            messagebox.showerror("Error", 
                                f"Need at least {self.num_components} students with complete data. " +
                                f"Currently only have {len(A)}.")
            return
        
        try:
            # Solve the system using least squares
            weights, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if weights sum to approximately 1
            weights_sum = np.sum(weights)
            normalized_weights = weights / weights_sum if abs(weights_sum) > 1e-10 else weights
            
            # Store the results
            self.data_matrix = A
            self.final_grades = b
            self.weights = normalized_weights
            
            # Display the results
            self.display_results()
            
            # Switch to results tab
            self.notebook.select(self.results_tab)
            
        except np.linalg.LinAlgError:
            messagebox.showerror("Error", "Could not solve the system. Please check your data.")

    def setup_results_tab(self):
        # This will be populated when results are ready
        self.results_frame = ttk.Frame(self.results_tab)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(self.results_frame, text="Calculate weights to see results").pack(pady=20)

    def display_results(self):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create a frame for the weights table
        table_frame = ttk.LabelFrame(self.results_frame, text="Component Weights")
        table_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Display the weights in a table
        for i, (component, weight) in enumerate(zip(self.component_names, self.weights)):
            frame = ttk.Frame(table_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=component, width=20, anchor="w").pack(side=tk.LEFT, padx=5)
            ttk.Label(frame, text=f"{weight:.4f} ({weight*100:.2f}%)", width=20).pack(side=tk.LEFT, padx=5)
        
        # Add summary
        summary_frame = ttk.LabelFrame(self.results_frame, text="Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Calculate Mean Absolute Error
        predicted_grades = np.dot(self.data_matrix, self.weights)
        mae = np.mean(np.abs(predicted_grades - self.final_grades))
        
        ttk.Label(summary_frame, text=f"Mean Absolute Error: {mae:.4f}").pack(anchor="w", padx=5, pady=2)
        ttk.Label(summary_frame, text=f"Sum of weights: {np.sum(self.weights):.4f}").pack(anchor="w", padx=5, pady=2)
        
        # Create visualization frame
        viz_frame = ttk.LabelFrame(self.results_frame, text="Visualizations")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs for different visualizations
        viz_notebook = ttk.Notebook(viz_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pie chart tab
        pie_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(pie_tab, text="Weight Distribution")
        
        # Scatter plot tab
        scatter_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(scatter_tab, text="Actual vs Predicted")
        
        # Create and display pie chart
        self.create_pie_chart(pie_tab)
        
        # Create and display scatter plot
        self.create_scatter_plot(scatter_tab)
        
        # Add export button
        ttk.Button(self.results_frame, text="Export Results", command=self.export_results).pack(pady=10)

    def create_pie_chart(self, parent):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(self.weights, labels=self.component_names, autopct='%1.1f%%', 
               startangle=90, shadow=True)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_scatter_plot(self, parent):
        predicted_grades = np.dot(self.data_matrix, self.weights)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(self.final_grades, predicted_grades)
        
        # Add perfect prediction line
        min_val = min(min(self.final_grades), min(predicted_grades))
        max_val = max(max(self.final_grades), max(predicted_grades))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('Actual Grades')
        ax.set_ylabel('Predicted Grades')
        ax.set_title('Actual vs Predicted Grades')
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_results(self):
        filename = simpledialog.askstring("Export", "Enter filename (without extension):", 
                                         parent=self.root)
        if not filename:
            return
        
        try:
            with open(f"{filename}.txt", "w") as f:
                f.write("Grade Component Weights\n")
                f.write("======================\n\n")
                
                for component, weight in zip(self.component_names, self.weights):
                    f.write(f"{component}: {weight:.4f} ({weight*100:.2f}%)\n")
                
                f.write("\nSummary\n")
                f.write("=======\n")
                predicted_grades = np.dot(self.data_matrix, self.weights)
                mae = np.mean(np.abs(predicted_grades - self.final_grades))
                f.write(f"Mean Absolute Error: {mae:.4f}\n")
                f.write(f"Sum of weights: {np.sum(self.weights):.4f}\n")
            
            messagebox.showinfo("Success", f"Results exported to {filename}.txt")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")

def main():
    root = tk.Tk()
    app = StudentGradeWeightCalculator(root)
    root.mainloop()

if __name__ == "__main__":
    main()