import os
from tkinter import *
from tkinter import filedialog, messagebox

from PIL import ImageTk, Image
from functions import *

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.jfif')


class ImageViewer:
    def __init__(self, directory):
        self.directory = directory
        self.image_files = [f for f in os.listdir(directory) if f.endswith(IMAGE_EXTENSIONS)]
        if not self.image_files:
            messagebox.showerror("Error", "No supported images found in the selected directory.")
            return

        self.List_images = [None] * len(self.image_files)
        self.List_photoimages = [None] * len(self.image_files)
        self.List_thumbnails = [None] * len(self.image_files)
        self.List_thumbnail_images = [None] * len(self.image_files)
        self.img_no = 0
        self.zoom_percent = 100
        self.is_fullscreen = False
        self.thumbnail_placeholder = ImageTk.PhotoImage(Image.new("RGB", (50, 50), "gray"))  # Grey placeholder
        self.loaded_thumbnails = set()
        self.status_bar = None

        # Preload current, previous and next images and thumbnails
        self.load_image_at_index(self.img_no)
        if self.img_no > 0:
            self.load_image_at_index(self.img_no - 1)
        if self.img_no < len(self.image_files) - 1:
            self.load_image_at_index(self.img_no + 1)

        self.init_window()

    def init_window(self):
        self.image_window = Toplevel()
        self.image_window.title("Image Viewer")
        self.create_menu()
        self.create_image_frame()
        self.create_thumbnail_frame()
        # self.create_zoom_controls()
        # self.create_navigation_buttons()
        self.status_bar = Label(self.image_window, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.grid(row=2, column=0, columnspan=4, sticky='ew')
        ...

        self.controls_frame = Frame(self.image_window)
        self.controls_frame.grid(row=1, column=1, columnspan=3, sticky='ew')

        self.create_zoom_controls(self.controls_frame)
        self.create_navigation_buttons(self.controls_frame)

        self.update_buttons()
        self.update_listbox()

        self.image_window.bind('<Right>', self.forward)
        self.image_window.bind('<Left>', self.back)
        self.image_window.bind("<F10>", self.toggle_fullscreen)
        self.image_window.bind("<Escape>", self.end_fullscreen)

        # Set weight to adjust canvas with window resize
        self.image_window.grid_rowconfigure(1, weight=1)
        self.image_window.grid_columnconfigure(1, weight=1)

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.image_window.attributes("-fullscreen", self.is_fullscreen)
        return "break"

    def end_fullscreen(self, event=None):
        self.is_fullscreen = False
        self.image_window.attributes("-fullscreen", False)
        return "break"

    def create_menu(self):
        menu_bar = Menu(self.image_window)
        self.image_window.config(menu=menu_bar)
        control_menu = Menu(menu_bar, tearoff=0)
        control_menu.add_command(label="Back", command=self.back)
        control_menu.add_command(label="Forward", command=self.forward)
        control_menu.add_command(label="Zoom In", command=lambda: self.adjust_zoom(10))
        control_menu.add_command(label="Zoom Out", command=lambda: self.adjust_zoom(-10))
        scale_menu = Menu(control_menu, tearoff=0)
        self.scale = Scale(scale_menu, from_=10, to=400, orient=HORIZONTAL)
        self.scale.set(100)
        self.scale.bind('<Motion>', lambda e: self.resize_image(self.scale.get()))
        self.scale.pack()
        control_menu.add_cascade(label="Zoom", menu=scale_menu)
        control_menu.add_command(label="Exit", command=self.image_window.quit)
        menu_bar.add_cascade(label="Controls", menu=control_menu)

    def create_image_frame(self):
        frame_image = Frame(self.image_window, width=700, height=400)
        frame_image.grid(row=0, column=1, columnspan=3, sticky='nsew')
        self.canvas_image = Canvas(frame_image, width=700, height=400)
        self.canvas_image.grid(row=0, column=0, sticky='nsew')
        x_scrollbar = Scrollbar(frame_image, orient="horizontal", command=self.canvas_image.xview)
        x_scrollbar.grid(row=1, column=0, sticky='ew')
        y_scrollbar = Scrollbar(frame_image, orient="vertical", command=self.canvas_image.yview)
        y_scrollbar.grid(row=0, column=1, sticky='ns')
        self.canvas_image.config(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        self.image_on_canvas = self.canvas_image.create_image(0, 0, anchor='nw',
                                                              image=self.List_photoimages[self.img_no])

        # Set weight for frame_image to adjust canvas size
        frame_image.grid_rowconfigure(0, weight=1)
        frame_image.grid_columnconfigure(0, weight=1)

    def create_thumbnail_frame(self):
        frame_thumbnails_container = Frame(self.image_window, width=70)  # Adjust width
        frame_thumbnails_container.grid(row=0, column=0, rowspan=3, sticky="ns")
        frame_thumbnails_container.grid_propagate(False)  # Forbid internal components to change size
        self.canvas_thumbnails = Canvas(frame_thumbnails_container, width=70, height=600)
        self.canvas_thumbnails.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar = Scrollbar(frame_thumbnails_container, orient="vertical")
        scrollbar.config(command=self.on_scroll)

        scrollbar.pack(side=RIGHT, fill=Y)
        self.canvas_thumbnails.config(yscrollcommand=scrollbar.set)
        self.frame_thumbnails = Frame(self.canvas_thumbnails, width=60,
                                      height=len(self.List_thumbnail_images) * 60)  # Increase height
        self.canvas_thumbnails.create_window((0, 0), window=self.frame_thumbnails, anchor='nw')

        for idx in range(len(self.image_files)):
            thumbnail_button = Button(self.frame_thumbnails, image=self.thumbnail_placeholder, relief=FLAT,
                                      command=lambda i=idx: self.on_select(i))
            thumbnail_button.pack(side=TOP)

        self.canvas_thumbnails.config(
            scrollregion=(0, 0, 60, len(self.List_thumbnail_images) * 60))  # Resize the scrolling area
        self.canvas_thumbnails.bind('<Configure>', self.load_visible_thumbnails)
        self.canvas_thumbnails.bind('<Enter>', self.load_visible_thumbnails)

    def on_scroll(self, *args):
        # Default scrolling behavior
        self.canvas_thumbnails.yview(*args)

        # Load visible thumbnails after scrolling
        self.load_visible_thumbnails()

    def load_visible_thumbnails(self, event=None):
        # Get the scroll position
        top = self.canvas_thumbnails.canvasy(0)
        height = self.canvas_thumbnails.winfo_height()

        # Calculate the index range of thumbnails that should be loaded
        start_idx = max(int(top // 60) - 2, 0)
        end_idx = min(int((top + height) // 60) + 2, len(self.image_files))

        # Load and update thumbnails for visible index ranges
        for idx in range(start_idx, end_idx):

            if idx not in self.loaded_thumbnails:
                thumbnail = Image.open(os.path.join(self.directory, self.image_files[idx])).resize((50, 50))
                thumbnail_img = ImageTk.PhotoImage(thumbnail)
                # Update button image
                self.frame_thumbnails.winfo_children()[idx].config(image=thumbnail_img)
                self.frame_thumbnails.winfo_children()[idx].image = thumbnail_img  # Keep reference
                self.loaded_thumbnails.add(idx)
            if self.List_images[idx] is None:
                self.load_image_at_index(idx)

    def create_zoom_controls(self, frame):
        button_zoom_in = Button(frame, text="Zoom In", command=lambda: self.adjust_zoom(10))
        button_zoom_in.grid(row=0, column=0, sticky='w')

        self.scale = Scale(frame, from_=10, to=400, orient=HORIZONTAL)
        self.scale.set(100)
        self.scale.grid(row=0, column=1, sticky='ew')
        self.scale.bind('<ButtonRelease-1>', lambda e: self.resize_image(self.scale.get()))

        button_zoom_out = Button(frame, text="Zoom Out", command=lambda: self.adjust_zoom(-10))
        button_zoom_out.grid(row=0, column=2, sticky='e')

    def create_navigation_buttons(self, frame):
        self.button_back = Button(frame, text="<<", command=self.back)
        self.button_back.grid(row=1, column=0, sticky='w')

        spacer = Label(frame, text=" " * 20)
        spacer.grid(row=1, column=1, sticky='ew')

        self.button_forward = Button(frame, text=">>", command=self.forward)
        self.button_forward.grid(row=1, column=2, sticky='e')

    def load_image_at_index(self, idx):
        """Load the image and thumbnail of the specified index."""

        try:
            # Load main image
            img = Image.open(os.path.join(self.directory, self.image_files[idx]))
            self.List_images[idx] = img
            self.List_photoimages[idx] = ImageTk.PhotoImage(img)

            # Load thumbnail if not loaded
            if self.List_thumbnails[idx] is None:
                thumb = img.resize((50, 50))
                self.List_thumbnails[idx] = thumb
                self.List_thumbnail_images[idx] = ImageTk.PhotoImage(thumb)

        except Exception as e:
            # your existing code for handling exception...

            # If there's an error, show a message and remove the problematic image from the list
            messagebox.showerror("Error", f"An error occurred while loading {self.image_files[idx]}: {str(e)}")
            self.image_files.pop(idx)

            # Recalculate the lists based on the updated image_files list
            self.List_images = [None] * len(self.image_files)
            self.List_photoimages = [None] * len(self.image_files)
            self.List_thumbnails = [None] * len(self.image_files)
            self.List_thumbnail_images = [None] * len(self.image_files)

    def forward(self, event=None):
        if self.img_no >= len(self.List_images) - 1:  # Return if it is the last image
            return
        self.img_no += 1
        if self.img_no < len(self.image_files) - 1:
            self.load_image_at_index(self.img_no + 1)
        self.update_image()
        self.update_buttons()
        self.load_visible_thumbnails()  # Load visible thumbnails
        self.update_status_bar()

    def back(self, event=None):
        if self.img_no <= 0:  # Return if it is the first image
            return
        self.img_no -= 1
        if self.img_no > 0:
            self.load_image_at_index(self.img_no - 1)
        self.update_image()
        self.update_buttons()
        self.load_visible_thumbnails()  # load visible thumbnails
        self.update_status_bar()

    def resize_image(self, percent):
        self.zoom_percent = percent
        img_resized = self.List_images[self.img_no].resize((int(self.List_images[self.img_no].width * percent / 100),
                                                            int(self.List_images[self.img_no].height * percent / 100)))
        self.List_photoimages[self.img_no] = ImageTk.PhotoImage(img_resized)
        self.canvas_image.itemconfig(self.image_on_canvas, image=self.List_photoimages[self.img_no])

    def update_image(self):
        self.canvas_image.itemconfig(self.image_on_canvas, image=self.List_photoimages[self.img_no])
        self.canvas_image.config(scrollregion=self.canvas_image.bbox(ALL))
        self.resize_image(self.zoom_percent)
        self.update_listbox()

    def update_buttons(self):
        self.button_back.config(state=NORMAL if self.img_no > 0 else DISABLED)
        self.button_forward.config(state=NORMAL if self.img_no < len(self.List_images) - 1 else DISABLED)

    def adjust_zoom(self, delta):
        new_zoom = self.zoom_percent + delta
        if 10 <= new_zoom <= 400:
            self.zoom_percent = new_zoom
            self.resize_image(new_zoom)
            self.scale.set(new_zoom)

    def on_select(self, idx):
        self.img_no = idx
        self.update_image()
        self.update_buttons()
        self.update_status_bar()

    def update_status_bar(self):
        file_path = os.path.join(self.directory, self.image_files[self.img_no])
        image_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        img = self.List_images[self.img_no]
        info_text = f"File: {self.image_files[self.img_no]}  |  Resolution: {img.width}x{img.height}  |  Size: {image_size:.2f} MB"
        self.status_bar.config(text=info_text)

    def update_listbox(self):
        for widget in self.frame_thumbnails.winfo_children():
            widget.config(relief=FLAT)
        self.frame_thumbnails.winfo_children()[self.img_no].config(relief=SOLID)
        self.canvas_thumbnails.yview_scroll(self.img_no - int(self.canvas_thumbnails.winfo_height() / 60), 'units')


def choose_directory():
    try:
        directory = filedialog.askdirectory()
        if directory:
            ImageViewer(directory)
    except FileNotFoundError:
        messagebox.showerror("Error", "The image file was not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


root = Tk()
root.title("Main Menu")
choose_button = Button(root, text="Choose Directory", command=choose_directory)
choose_button.pack()
root.mainloop()