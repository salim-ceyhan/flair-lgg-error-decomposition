from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .analysis_utils import save_histogram_analysis
from .flow_utils import default_output_path, load_grayscale_image, save_result_image
from .run_metric import METRIC_DEFAULTS, run_metric


class MetricUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('FAC Metric Runner')
        self.root.geometry('860x720')
        self.root.minsize(820, 680)

        self.metric_var = tk.StringVar(value='newmetric')
        self.image_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.output_subdir_var = tk.StringVar(value='Filtered')
        self.analysis_subdir_var = tk.StringVar(value='Analysis')
        self.beta_var = tk.StringVar()
        self.dt_var = tk.StringVar()
        self.iterations_var = tk.StringVar()
        self.save_analysis_var = tk.BooleanVar(value=False)
        self.timestamp_subdir_var = tk.BooleanVar(value=False)
        self.show_result_var = tk.BooleanVar(value=True)

        self.output_auto_managed = True
        self._updating_output_var = False
        self._timestamp_suffix: str | None = None

        self.run_button: ttk.Button | None = None
        self.status_text: tk.Text | None = None
        self.beta_entry: ttk.Spinbox | None = None
        self.dt_entry: ttk.Spinbox | None = None
        self.iterations_entry: ttk.Spinbox | None = None
        self.output_entry: ttk.Entry | None = None

        self._build_ui()
        self._wire_events()
        self._apply_metric_defaults()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=(18, 16))
        main.pack(fill='both', expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        hero = ttk.Frame(main, style='Card.TFrame', padding=(18, 16))
        hero.grid(row=0, column=0, sticky='ew', pady=(0, 14))
        hero.columnconfigure(0, weight=1)

        ttk.Label(
            hero,
            text='Beltrami / Finsler Metric Runner',
            style='Title.TLabel',
        ).grid(row=0, column=0, sticky='w')
        ttk.Label(
            hero,
            text='Pick an image, tune the metric parameters, and save the filtered result with optional histogram analysis.',
            style='Subtitle.TLabel',
            wraplength=760,
        ).grid(row=1, column=0, sticky='w', pady=(6, 0))

        top_grid = ttk.Frame(main)
        top_grid.grid(row=1, column=0, sticky='nsew')
        top_grid.columnconfigure(0, weight=3)
        top_grid.columnconfigure(1, weight=2)

        left_col = ttk.Frame(top_grid)
        left_col.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        left_col.columnconfigure(0, weight=1)

        right_col = ttk.Frame(top_grid)
        right_col.grid(row=0, column=1, sticky='nsew')
        right_col.columnconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(left_col, text='Input', padding=14)
        input_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        self._add_label(input_frame, 0, 'Method')
        metric_combo = ttk.Combobox(
            input_frame,
            textvariable=self.metric_var,
            values=sorted(METRIC_DEFAULTS.keys()),
            state='readonly',
        )
        metric_combo.grid(row=0, column=1, sticky='ew', padx=(10, 0), pady=4)
        metric_combo.bind('<<ComboboxSelected>>', lambda _event: self._on_metric_changed())

        self._add_label(input_frame, 1, 'Image')
        image_frame = ttk.Frame(input_frame)
        image_frame.grid(row=1, column=1, sticky='ew', padx=(10, 0), pady=4)
        image_frame.columnconfigure(0, weight=1)
        ttk.Entry(image_frame, textvariable=self.image_var).grid(row=0, column=0, sticky='ew')
        ttk.Button(image_frame, text='Browse Image', command=self._browse_image, style='Accent.TButton').grid(
            row=0,
            column=1,
            padx=(8, 0),
        )

        params_frame = ttk.LabelFrame(left_col, text='Parameters', padding=14)
        params_frame.grid(row=1, column=0, sticky='ew')
        params_frame.columnconfigure(1, weight=1)

        self._add_label(params_frame, 0, 'Beta')
        self.beta_entry = ttk.Spinbox(
            params_frame,
            textvariable=self.beta_var,
            from_=0.0,
            to=10.0,
            increment=0.1,
            format='%.2f',
        )
        self.beta_entry.grid(row=0, column=1, sticky='ew', padx=(10, 0), pady=4)

        self._add_label(params_frame, 1, 'dt')
        self.dt_entry = ttk.Spinbox(
            params_frame,
            textvariable=self.dt_var,
            from_=0.0,
            to=10.0,
            increment=0.01,
            format='%.3f',
        )
        self.dt_entry.grid(row=1, column=1, sticky='ew', padx=(10, 0), pady=4)

        self._add_label(params_frame, 2, 'Iterations')
        self.iterations_entry = ttk.Spinbox(
            params_frame,
            textvariable=self.iterations_var,
            from_=1,
            to=1000,
            increment=1,
        )
        self.iterations_entry.grid(row=2, column=1, sticky='ew', padx=(10, 0), pady=4)

        output_frame = ttk.LabelFrame(right_col, text='Output', padding=14)
        output_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)

        ttk.Label(output_frame, text='Output Path', style='FieldLabel.TLabel').grid(
            row=0,
            column=0,
            sticky='w',
            pady=(0, 4),
        )
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=0, sticky='ew')
        ttk.Button(output_frame, text='Browse Output', command=self._browse_output).grid(
            row=1,
            column=1,
            padx=(8, 0),
        )
        ttk.Button(output_frame, text='Use Default Output', command=self._fill_default_output).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky='ew',
            pady=(10, 0),
        )

        subdir_frame = ttk.Frame(output_frame)
        subdir_frame.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(12, 0))
        subdir_frame.columnconfigure(1, weight=1)

        self._add_label(subdir_frame, 0, 'Output Subdir')
        ttk.Entry(subdir_frame, textvariable=self.output_subdir_var).grid(
            row=0,
            column=1,
            sticky='ew',
            padx=(10, 0),
            pady=4,
        )

        self._add_label(subdir_frame, 1, 'Analysis Subdir')
        ttk.Entry(subdir_frame, textvariable=self.analysis_subdir_var).grid(
            row=1,
            column=1,
            sticky='ew',
            padx=(10, 0),
            pady=4,
        )

        options = ttk.LabelFrame(right_col, text='Options', padding=14)
        options.grid(row=1, column=0, sticky='ew')
        options.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            options,
            text='Save histogram analysis',
            variable=self.save_analysis_var,
        ).grid(row=0, column=0, sticky='w', pady=3)
        ttk.Checkbutton(
            options,
            text='Append timestamp to output subdir',
            variable=self.timestamp_subdir_var,
            command=self._handle_output_config_change,
        ).grid(row=1, column=0, sticky='w', pady=3)
        ttk.Checkbutton(
            options,
            text='Open saved result image',
            variable=self.show_result_var,
        ).grid(row=2, column=0, sticky='w', pady=3)

        actions = ttk.Frame(main, padding=(0, 12, 0, 10))
        actions.grid(row=2, column=0, sticky='ew')
        self.run_button = ttk.Button(actions, text='Run Filter', command=self._start_run, style='Accent.TButton')
        self.run_button.pack(side='left')
        ttk.Button(actions, text='Clear Log', command=self._clear_log).pack(side='left', padx=(8, 0))

        log_frame = ttk.LabelFrame(main, text='Log', padding=10)
        log_frame.grid(row=3, column=0, sticky='nsew')
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.status_text = tk.Text(
            log_frame,
            height=16,
            wrap='word',
            relief='flat',
            bg='#fbfcfe',
            fg='#1d2733',
            padx=8,
            pady=8,
        )
        self.status_text.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.status_text.configure(yscrollcommand=scrollbar.set)

    def _wire_events(self) -> None:
        self.image_var.trace_add('write', lambda *_args: self._handle_image_change())
        self.output_subdir_var.trace_add('write', lambda *_args: self._handle_output_config_change())
        if self.output_entry is not None:
            self.output_entry.bind('<KeyRelease>', self._mark_output_manual)

    @staticmethod
    def _add_label(parent: ttk.Frame, row: int, text: str) -> None:
        ttk.Label(parent, text=text, style='FieldLabel.TLabel').grid(row=row, column=0, sticky='w', pady=4)

    def _browse_image(self) -> None:
        selected = filedialog.askopenfilename(
            title='Select image',
            filetypes=[
                ('Image files', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'),
                ('All files', '*.*'),
            ],
        )
        if selected:
            self.image_var.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title='Select output image path',
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('All files', '*.*')],
        )
        if selected:
            self.output_auto_managed = False
            self._set_output_value(selected)

    def _on_metric_changed(self) -> None:
        self._apply_metric_defaults()
        self._handle_output_config_change()

    def _apply_metric_defaults(self) -> None:
        metric = self.metric_var.get()
        defaults = METRIC_DEFAULTS[metric]
        beta = defaults['beta']
        self.beta_var.set('' if beta is None else f'{beta:.2f}')
        self.dt_var.set(f"{defaults['dt']:.3f}")
        self.iterations_var.set(str(defaults['iterations']))
        self._sync_beta_state()

    def _sync_beta_state(self) -> None:
        if self.beta_entry is None:
            return
        state = 'disabled' if self.metric_var.get() == 'beltrami' else 'normal'
        self.beta_entry.configure(state=state)

    def _handle_image_change(self) -> None:
        image_path_str = self.image_var.get().strip()
        if not image_path_str:
            if self.output_auto_managed:
                self._set_output_value('')
            return
        self._update_default_output_if_needed()

    def _handle_output_config_change(self) -> None:
        if self.timestamp_subdir_var.get():
            if self._timestamp_suffix is None:
                self._timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self._timestamp_suffix = None
        self._update_default_output_if_needed()

    def _mark_output_manual(self, _event: tk.Event[tk.Misc]) -> None:
        if self._updating_output_var:
            return
        self.output_auto_managed = False

    def _set_output_value(self, value: str) -> None:
        self._updating_output_var = True
        try:
            self.output_var.set(value)
        finally:
            self._updating_output_var = False

    def _update_default_output_if_needed(self) -> None:
        if not self.output_auto_managed:
            return
        image_path_str = self.image_var.get().strip()
        if not image_path_str:
            return
        output_path = default_output_path(
            Path(image_path_str),
            self.metric_var.get(),
            output_subdir=self._resolved_output_subdir(),
        )
        self._set_output_value(str(output_path))

    def _fill_default_output(self) -> None:
        image_path_str = self.image_var.get().strip()
        if not image_path_str:
            messagebox.showinfo('Output Path', 'Select an image first.')
            return

        self.output_auto_managed = True
        if self.timestamp_subdir_var.get():
            self._timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._update_default_output_if_needed()

    def _resolved_output_subdir(self) -> str:
        output_subdir = self.output_subdir_var.get().strip() or 'Filtered'
        if self.timestamp_subdir_var.get():
            if self._timestamp_suffix is None:
                self._timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_subdir = f'{output_subdir}_{self._timestamp_suffix}'
        return output_subdir

    def _clear_log(self) -> None:
        if self.status_text is not None:
            self.status_text.delete('1.0', tk.END)

    def _log(self, message: str) -> None:
        if self.status_text is None:
            return
        self.status_text.insert(tk.END, message + '\n')
        self.status_text.see(tk.END)

    def _start_run(self) -> None:
        image_path_str = self.image_var.get().strip()
        if not image_path_str:
            messagebox.showerror('Missing Image', 'Please select an input image.')
            return

        try:
            metric = self.metric_var.get()
            beta = None if metric == 'beltrami' else float(self.beta_var.get())
            dt = float(self.dt_var.get())
            iterations = int(float(self.iterations_var.get()))
            image_path = Path(image_path_str)

            output_path_str = self.output_var.get().strip()
            if output_path_str:
                output_path = Path(output_path_str)
            else:
                output_path = default_output_path(
                    image_path,
                    metric,
                    output_subdir=self._resolved_output_subdir(),
                )
                self.output_auto_managed = True
                self._set_output_value(str(output_path))
        except ValueError as exc:
            messagebox.showerror('Invalid Parameters', f'Check parameter values.\n\n{exc}')
            return

        if self.run_button is not None:
            self.run_button.configure(state='disabled')
        self._log('Running filter...')

        thread = threading.Thread(
            target=self._run_filter,
            args=(metric, image_path, beta, dt, iterations, output_path),
            daemon=True,
        )
        thread.start()

    def _run_filter(
        self,
        metric: str,
        image_path: Path,
        beta: float | None,
        dt: float,
        iterations: int,
        output_path: Path,
    ) -> None:
        try:
            image = load_grayscale_image(image_path)
            if image is None:
                raise FileNotFoundError(f'Image not found: {image_path}')

            start = datetime.now()
            result = run_metric(metric, image, beta, dt, iterations)
            elapsed = (datetime.now() - start).total_seconds()

            save_result_image(result, output_path, show=self.show_result_var.get())

            analysis_path = None
            if self.save_analysis_var.get():
                analysis_subdir = self.analysis_subdir_var.get().strip() or 'Analysis'
                analysis_path = output_path.parent / analysis_subdir / f'{output_path.stem}_analysis.png'
                save_histogram_analysis(image, result, analysis_path)

            self.root.after(
                0,
                lambda: self._finish_run(
                    metric,
                    image_path,
                    output_path,
                    analysis_path,
                    beta,
                    dt,
                    iterations,
                    elapsed,
                ),
            )
        except Exception:
            error_text = traceback.format_exc()
            self.root.after(0, lambda: self._fail_run(error_text))

    def _finish_run(
        self,
        metric: str,
        image_path: Path,
        output_path: Path,
        analysis_path: Path | None,
        beta: float | None,
        dt: float,
        iterations: int,
        elapsed: float,
    ) -> None:
        if self.run_button is not None:
            self.run_button.configure(state='normal')

        self._log(f'Metric: {metric}')
        self._log(f'Input: {image_path}')
        self._log(f'Output: {output_path}')
        if analysis_path is not None:
            self._log(f'Analysis: {analysis_path}')
        if beta is not None:
            self._log(f'beta={beta}, dt={dt}, iterations={iterations}')
        else:
            self._log(f'dt={dt}, iterations={iterations}')
        self._log(f'Elapsed time: {elapsed:.3f}s')
        self._log('Done.')
        messagebox.showinfo('Completed', f'Filtering completed.\n\nSaved to:\n{output_path}')

    def _fail_run(self, error_text: str) -> None:
        if self.run_button is not None:
            self.run_button.configure(state='normal')
        self._log('Error:')
        self._log(error_text)
        messagebox.showerror('Run Failed', error_text)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if 'vista' in style.theme_names():
        style.theme_use('vista')

    style.configure('Title.TLabel', font=('Segoe UI Semibold', 17))
    style.configure('Subtitle.TLabel', font=('Segoe UI', 10), foreground='#4f6475')
    style.configure('FieldLabel.TLabel', font=('Segoe UI Semibold', 10))
    style.configure('TLabelframe', padding=2)
    style.configure('TLabelframe.Label', font=('Segoe UI Semibold', 10))
    style.configure('Accent.TButton', padding=(12, 7))
    style.configure('Card.TFrame')

    MetricUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
