import json
import platform
import sys
import threading
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from tkinter.messagebox import askyesno
import numpy as np

from PIL import ImageTk, Image
sys.path.append('./')
from source.main import main_gui
import warnings

warnings.filterwarnings("ignore")

CONFIG_FILE = "settings.json"

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.jfif', '.gif')

ENTROPY_METHODS = [
    'lbp',
    'lbp_gabor',
    'adapt',
    'RGBCM'
]

COLOR_OPTIONS = ['rgb', 'hsb', 'YCbCr', 'greyscale']

LIMIT_IMAGES = 500 # Limit the number of images to load in one page of thumbnail frame

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


class IORedirector(object):
    def __init__(self, text_area):
        self.text_area = text_area

    def write(self, str_):
        self.text_area.insert(END, str_ + '\n')
        self.text_area.see(END)

    def flush(self):
        pass


class ClassViewer(object):
    def __init__(self, directory):
        self.directory = directory
        self.img_dict = self.get_image_paths_from_subfolders(directory)
        self.init_window()

    
    def init_window(self):
        self.image_window = Toplevel()
        self.image_window.title("Image Viewer")
        self.image_window.protocol('WM_DELETE_WINDOW', lambda: self.thread_it(self.clos_window))
        self.image_window.geometry()  # The desired initial size
        self.create_folder_name_frame()
        self.init_load_images(self)


    def clos_window(self):
        ans = askyesno(title='WARNING', message='Are you sure to close the window?')
        if ans:
            self.image_window.destroy()
            sys.exit()
        else:
            return None
        

    def create_folder_name_frame():
        pass
    
    
    def get_image_paths_from_subfolders(directory):
        """Return a dictionary with subfolder names as keys and lists of image paths as values."""
        valid_image_extensions = IMAGE_EXTENSIONS
        image_dict = {}

        for subdir, _, files in os.walk(directory):
            if subdir == directory:  # skip the main directory
                continue
            image_paths = [os.path.join(subdir, file) for file in files if os.path.splitext(file)[1].lower() in valid_image_extensions]
            if image_paths:  # only add to dictionary if there are valid images in the subfolder
                folder_name = os.path.basename(subdir)
                image_dict[folder_name] = image_paths

        return image_dict

    
    def init_load_images(self):
        for folder_name, image_paths in self.img_dict.items():
            self.init_load_folder_name(self)
            for path in image_paths:
                print(f"Image Path: {path}")

    
    def init_load_folder_name(self):
        pass

    
    def thread_it(self, func, *args):
        """ Pack functions into threads """
        self.myThread = threading.Thread(target=func, args=args)
        self.myThread.daemon = True  # When the main thread exits, the sub-threads will follow and exit directly, regardless of whether the operation is completed or not.
        self.myThread.start()


class ImageViewer:
    def __init__(self, directory):
        self.directory = directory
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        self.default_save_directory = self.load_default_directory()

        self.np_array = self.load_all_images_to_np_arrays(directory)

        self.os_name = platform.system()
        self.img_ent_data = None
        self.initialize_all_images()
        self.img_no = 0
        self.sorted_indices = None
        self.json_data = None
        self.prob_data = []
        self.zoom_percent = 100
        self.is_fullscreen = False
        self.thumbnail_placeholder = ImageTk.PhotoImage(Image.new("RGB", (60, 60), "gray"))  # Grey placeholder
        self.loaded_thumbnails = set()
        self.status_bar = None
        self.entropies_calculated = 0

        # Preload current, previous and next images and thumbnails
        self.load_image_at_index(self.img_no)
        if self.img_no > 0:
            self.load_image_at_index(self.img_no - 1)
        if self.img_no < len(self.np_array) - 1:
            self.load_image_at_index(self.img_no + 1)

        self.init_window()
        self.console = None

    
    def init_window(self):
        self.image_window = Toplevel()
        self.image_window.title("Image Viewer")
        self.image_window.protocol('WM_DELETE_WINDOW', lambda: self.thread_it(self.clos_window))
        self.image_window.geometry(self.center_window_coordinates(600, 632))  # The desired initial size
        self.create_menu()
        self.create_image_frame()
        self.create_thumbnail_frame()

        self.status_bar = Label(self.image_window, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.grid(row=3, column=0, columnspan=4, sticky='ew')
        ...
        #self.console = Text(self.image_window, height=10, width=50)
        #self.console.grid(row=3, column=0, columnspan=2, pady=20, padx=10, sticky='ew')

        self.controls_frame = Frame(self.image_window)
        self.controls_frame.grid(row=1, column=1, columnspan=4, sticky='ew')

        self.create_zoom_controls(self.controls_frame)
        self.create_navigation_buttons(self.controls_frame)
        self.create_calculation_buttons(self.controls_frame)

        self.update_buttons()
        self.update_listbox()
        self.update_confirm_button_state()
        #self.button_save.config(state=DISABLED)# Disable save button until entropy calculation is complete

        self.image_window.bind('<Right>', self.forward)
        self.image_window.bind('<Left>', self.back)
        self.image_window.bind("<F10>", self.toggle_fullscreen)
        self.image_window.bind("<Escape>", self.end_fullscreen)

        # Set weight to adjust canvas with window resize
        self.image_window.grid_rowconfigure(1, weight=1)
        self.image_window.grid_columnconfigure(1, weight=1)
        self.controls_frame.grid_rowconfigure(1, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)

        #sys.stdout = IORedirector(self.console)

        self.center_window(self.image_window)

        self.forward()
        self.back()

    
    def adjust_zoom(self, delta):
        new_zoom = self.zoom_percent + delta
        if 10 <= new_zoom <= 400:
            self.zoom_percent = new_zoom
            self.resize_image(new_zoom)
            self.scale.set(new_zoom)
    
    
    def back(self, event=None):
        if self.img_no <= 0:  # Return if it is the first image
            return
        self.img_no -= 1
        if self.img_no > 0:
            self.load_image_at_index(self.img_no - 1)
        self.update_image()
        self.update_buttons()

        self.load_visible_thumbnails()  # load visible thumbnails
        self.scroll_to_img_no()  # Scroll to the current image
        self.update_status_bar()

    
    
    def center_window(self, window):
        window.update_idletasks()  # Ensure that the window size has been 'implemented'
    
        # Obtain the width and height of the screen
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
    
        # Obtain the width and height of the window
        width = window.winfo_width()
        height = window.winfo_height()

        # Calculate the x and y coordinates to center the window
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))


    def center_window_coordinates(self, width=None, height=None):
        if width is None:
            width = self.image_window.winfo_reqwidth()
        if height is None:
            height = self.image_window.winfo_reqheight()
        x = (self.image_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.image_window.winfo_screenheight() // 2) - (height // 2)
        return '{}x{}+{}+{}'.format(width, height, x, y)
    

    def choose_config_location(self):
        directory = filedialog.askdirectory()
        if directory:
            self.config_path = os.path.join(directory, "settings.json")
            # Save the current default save directory to a new location
            self.save_default_directory(self.default_save_directory)

    
    def choose_default_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.save_default_directory(directory)
            self.default_save_directory = directory
    
    
    def create_calculation_buttons(self, frame):
        method_label = Label(frame, text="Entropy Method")
        color_label = Label(frame, text="")
        method_label.grid(row=2, column=0, sticky='ew')
        color_label.grid(row=1, column=0, sticky='ew')
        # Create a Combobox and make it visible
        self.combo = ttk.Combobox(frame, values=ENTROPY_METHODS, state='readonly')
        self.combo.grid(row=2, column=1)  # Set the position of the combobox
        #self.combo_color = ttk.Combobox(frame, values=COLOR_OPTIONS, state='readonly')
        #self.combo_color.grid(row=1, column=1)  # Set the position of the combobox

        # Binding selection event
        self.combo.bind("<<ComboboxSelected>>", self.on_combo_select)
        #self.combo_color.bind("<<ComboboxSelected>>", self.on_combo_color_select)

        # The save button is arranged on the right side of the combobox
        #self.button_save = Button(frame, text="Save", command=self.save)
        #self.button_save.grid(row=2, column=3, sticky='ew')

        self.confirm_button = Button(frame, text="Confirm", command=lambda: self.thread_it(self.on_confirm_click))
        self.confirm_button.grid(row=2, column=2, sticky='ew')

 
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

        settings_menu = Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="Set Default Save Directory", command=self.choose_default_directory)
        settings_menu.add_command(label="Choose Config File Location", command=self.choose_config_location)
        menu_bar.add_cascade(label="Settings", menu=settings_menu)
    
    
    def create_navigation_buttons(self, frame):
        self.button_back = Button(frame, text="<<", command=self.back)
        self.button_back.grid(row=1, column=2, sticky='ew')

        spacer = Label(frame, text=" " * 20)
        spacer.grid(row=1, column=0, sticky='ew')

        self.button_forward = Button(frame, text=">>", command=self.forward)
        self.button_forward.grid(row=1, column=3, sticky='ew')


    def create_thumbnail_frame(self):
        frame_thumbnails_container = Frame(self.image_window, width=60)  # Adjust width
        frame_thumbnails_container.grid(row=0, column=0, rowspan=3, sticky="ns")
        frame_thumbnails_container.grid_propagate(False)  # Forbid internal components to change size
        self.canvas_thumbnails = Canvas(frame_thumbnails_container, width=60, height=600)
        self.canvas_thumbnails.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar = Scrollbar(frame_thumbnails_container, orient="vertical")
        scrollbar.config(command=self.on_scroll)

        if self.os_name == 'Darwin':
            gap = 8
        if self.os_name == 'Linux':
            gap = 6
        if self.os_name == 'Windows':
            gap = 6
        
        scrollbar.pack(side=RIGHT, fill=Y)
        self.canvas_thumbnails.config(yscrollcommand=scrollbar.set)
        self.frame_thumbnails = Frame(self.canvas_thumbnails, width=60,
                                      height=len(self.List_thumbnail_images) * (60+gap))  # Increase height
        self.canvas_thumbnails.create_window((0, 0), window=self.frame_thumbnails, anchor='nw')

        for idx in range(len(self.np_array)):
            thumbnail_button = Button(self.frame_thumbnails, image=self.thumbnail_placeholder, relief=FLAT, command=lambda i=idx: self.on_select(i))
            thumbnail_button.grid(row=idx, column=0, sticky='nsew', padx=0, pady=0, ipadx=0, ipady=0)
            self.frame_thumbnails.rowconfigure(idx, weight=1)


        self.canvas_thumbnails.config(
            scrollregion=(0, 0, 60, len(self.List_thumbnail_images) * (60+gap)))  # Resize the scrolling area
        self.canvas_thumbnails.bind('<Configure>', self.load_visible_thumbnails)
        self.canvas_thumbnails.bind('<Enter>', self.load_visible_thumbnails)

    
    def create_zoom_controls(self, frame):
        spacer = Label(frame, text=" " * 20)
        spacer.grid(row=0, column=0, sticky='ew')
        
        button_zoom_in = Button(frame, text="Zoom In", command=lambda: self.adjust_zoom(10))
        button_zoom_in.grid(row=0, column=2, sticky='ew')

        self.scale = Scale(frame, from_=10, to=400, orient=HORIZONTAL)
        self.scale.set(100)
        self.scale.grid(row=0, column=1, sticky='ew')
        self.scale.bind('<ButtonRelease-1>', lambda e: self.resize_image(self.scale.get()))

        button_zoom_out = Button(frame, text="Zoom Out", command=lambda: self.adjust_zoom(-10))
        button_zoom_out.grid(row=0, column=3, sticky='ew')
    
    
    def clos_window(self):
        ans = askyesno(title='WARNING', message='Are you sure to exit the program?\nIf yes exit, otherwise continue!')
        if ans:
            self.image_window.destroy()
            sys.exit()
        else:
            return None
        

    def entropy_calculation_complete(self, path):  # This method should be called once the entropy calculation is done
        self.load_json_entropy_data(path)
        self.rearrange_nparray()
        self.refresh_all_images(self.np_array)
        self.scroll_to_img_no()
        self.entropies_calculated = 1
        #self.button_save.config(state=NORMAL)
        self.confirm_button.config(state=NORMAL)
        self.forward()
        self.back()
    
    
    def end_fullscreen(self, event=None):
        self.is_fullscreen = False
        self.image_window.attributes("-fullscreen", False)
        return "break"
        
    
    def forward(self, event=None):
        if self.img_no >= len(self.np_array) - 1:  # Return if it is the last image
            return
        self.img_no += 1
        if self.img_no < len(self.np_array) - 1:
            self.load_image_at_index(self.img_no + 1)
        self.update_image()
        self.update_buttons()
        self.load_visible_thumbnails()  # Load visible thumbnails
        self.scroll_to_img_no()  # Scroll to the current image
        self.update_status_bar()
    

    def initialize_all_images(self):
        self.List_images = [None] * len(self.np_array)
        self.List_photoimages = [None] * len(self.np_array)
        self.List_thumbnails = [None] * len(self.np_array)
        self.List_thumbnail_images = [None] * len(self.np_array)

    
    def load_default_directory(self):
        try:
            with open(self.config_path, 'r') as file:
                data = json.load(file)
                return data.get('default_directory', '')
        except FileNotFoundError:
            return ''
    
    
    def load_all_images_to_np_arrays(self, directory):
        np_image_list = []

        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(IMAGE_EXTENSIONS):  # Add or modify extensions as needed
                    full_path = os.path.join(dirpath, filename)
                
                    # Load the image
                    img = Image.open(full_path)

                    # Ensure it's in RGB mode
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Convert to numpy array and append to list
                    np_img = np.array(img)
                    np_image_list.append(np_img)

        return np_image_list

    
    
    def load_image_at_index(self, idx):
        """Load the image and thumbnail of the specified index."""

        try:
            # Load main image
            img = Image.fromarray(self.np_array[idx])
            self.List_images[idx] = img
            self.List_photoimages[idx] = ImageTk.PhotoImage(img)

            # Load thumbnail if not loaded
            if self.List_thumbnails[idx] is None:
                thumb = img.resize((60, 60))
                self.List_thumbnails[idx] = thumb
                self.List_thumbnail_images[idx] = ImageTk.PhotoImage(thumb)

        except Exception as e:

            # If there's an error, show a message and remove the problematic image from the list
            messagebox.showerror("Error", f"An error occurred while loading {self.np_array[idx]}: {str(e)}")
            self.np_array.pop(idx)

            # Recalculate the lists based on the updated image_files list
            self.List_images = [None] * len(self.np_array)
            self.List_photoimages = [None] * len(self.np_array)
            self.List_thumbnails = [None] * len(self.np_array)
            self.List_thumbnail_images = [None] * len(self.np_array)
    
    
    def load_visible_thumbnails(self, event=None):
        # Get the scroll position
        top = self.canvas_thumbnails.canvasy(0)
        height = self.canvas_thumbnails.winfo_height()

        # Calculate the index range of thumbnails that should be loaded
        start_idx = max(int(top // 60) - 2, 0)
        end_idx = min(int((top + height) // 60) + 2, len(self.np_array))

        # Load and update thumbnails for visible index ranges
        for idx in range(start_idx, end_idx):

            if idx not in self.loaded_thumbnails:
                thumbnail = Image.fromarray(self.np_array[idx]).resize((60, 60))
                thumbnail_img = ImageTk.PhotoImage(thumbnail)
                # Update button image
                self.frame_thumbnails.winfo_children()[idx].config(image=thumbnail_img)
                self.frame_thumbnails.winfo_children()[idx].image = thumbnail_img  # Keep reference
                self.loaded_thumbnails.add(idx)
            if self.List_images[idx] is None:
                self.load_image_at_index(idx)
    
    
    def load_json_entropy_data(self, path):
        json_path = path + '/entropy_results.json'
        # Load the JSON data
        try:
            with open(json_path, 'r') as file:
                self.json_data = json.load(file)
        except FileNotFoundError:
            messagebox.showerror(f"Error: File not found at {json_path}")
        except json.JSONDecodeError:
            messagebox.showerror(f"Error: Unable to decode JSON from {json_path}")
        except Exception as e:
            messagebox.showerror(f"An unexpected error occurred: {e}")

    
    
    def on_combo_color_select(self, event=None):
        selected_color = self.combo.get()
        self.update_confirm_button_state()


    def on_combo_select(self, event=None):
        selected_item = self.combo.get()
        self.update_confirm_button_state()


    def on_confirm_click(self):
        self.image_window.after(0, self.start_preprocess)
        # The logic of sorting and displaying pictures based on the entropy method selected by combo box
        method = {self.combo.get(): None}
        #selected_color = self.combo_color.get()
        #self.start_entropy_calculation()
        save_directory = self.default_save_directory
        if not save_directory or not os.path.exists(save_directory):
            # If there's no default directory in the settings or it doesn't exist, ask the user
            folder_path = filedialog.askdirectory()
        else:
            folder_path = save_directory

        # Check if a valid directory was chosen or retrieved from the settings
        if not folder_path:
            return
        main_gui(folder_path, self.directory, method, None , 50*50, 1000, callback=self.update_preprogress, processed_level = 0)
        self.image_window.after(0, self.preprogress_window.destroy)
        self.image_window.after(0, self.entropy_calculation_complete, folder_path)
        #self.image_window.after(0, self.progress_window.destroy)
    
    
    def on_scroll(self, *args):
        # Check if the first argument is 'scroll'
        if args[0] == 'scroll':
            # Double the scroll speed by multiplying second argument
            modified_args = (args[0], int(args[1]) * 2, args[2])
            self.canvas_thumbnails.yview(*modified_args)
        else:
            # Default behavior
            self.canvas_thumbnails.yview(*args)

        # Load visible thumbnails after scrolling
        self.load_visible_thumbnails()



    def on_select(self, idx):
        self.img_no = idx
        self.load_visible_thumbnails()
        self.scroll_to_img_no()
        self.update_image()
        self.update_buttons()
        self.update_status_bar()
        
        if self.img_no != len(self.np_array) - 1:
            self.forward()
            self.back()


    def plot_bar_chart(self, frame):
        df = pd.DataFrame(self.prob_data[self.img_no])
        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(5, 4))
    
        # Plot data
        ax.bar(df['categories'], df['probabilities'], color='blue', alpha=0.7)
        ax.set_ylabel('Probabilities')
        ax.set_title('Probability distribution')
    
        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, padx=10, pady=10)
        canvas.draw()


    def rearrange_nparray(self):
        data = self.json_data
        self.entropy = [item['entropy_results'][0]['result'][0][0][0] for item in data]
        self.sorted_indices = list(np.argsort(self.entropy))
    
        # Rearrange the list based on sorted indices
        self.np_array = [self.np_array[i] for i in self.sorted_indices]

    
    def refresh_all_images(self, np_arrays):
        self.img_no=0
        self.initialize_all_images()
        # Ensure the length of numpy arrays matches the length of image lists
        if len(np_arrays) != len(self.List_images):
            messagebox.showerror("Mismatch in number of images and numpy arrays!")
            return

        for idx, np_array in enumerate(np_arrays):
            self.update_images_from_array(np_array, idx)

        self.update_image()
        self.update_buttons()
        self.update_status_bar()
        self.update_all_thumbnails()


    def resize_image(self, percent):
        if not self.List_images[self.img_no]:
            print("Image not loaded or None")
            return

        self.zoom_percent = percent
        img_resized = self.List_images[self.img_no].resize((int(self.List_images[self.img_no].width * percent / 100),
                                                            int(self.List_images[self.img_no].height * percent / 100)))
        self.List_photoimages[self.img_no] = ImageTk.PhotoImage(img_resized)
        self.canvas_image.itemconfig(self.image_on_canvas, image=self.List_photoimages[self.img_no])
    
    
    def save(self):
        # First, try to get the default save directory from settings.json
        default_save_directory = self.load_default_directory()

        if not default_save_directory or not os.path.exists(default_save_directory):
            # If there's no default directory in the settings or it doesn't exist, ask the user
            folder_path = filedialog.askdirectory()
        else:
            folder_path = default_save_directory

        # Check if a valid directory was chosen or retrieved from the settings
        if not folder_path:
            return

        images_arr = self.img_ent_data  # Retrieve your list/array of images here
        if images_arr is None:
            messagebox.showerror("Error", "Please complete an entropy sort first.")
            return

        # 1. Get the selected entropy method
        entropy_method = self.combo.get()
        #color = self.combo_color.get()

        # 2. Get the current time
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format: YYYYMMDD_HHMMSS

        # 3. Create a subfolder name based on entropy method and current time
        #subfolder_name = f"{color}_{entropy_method}_{current_time}"
        #subfolder_path = os.path.join(folder_path, subfolder_name)

        # 4. Create the subfolder
        #if not os.path.exists(subfolder_path):
            #os.makedirs(subfolder_path)

        # Save images to the subfolder
        #save_img(subfolder_path, images_arr)


    def save_default_directory(self, user_chosen_path):
        settings = {
            "default_directory": user_chosen_path
        }

        with open(self.config_path, "w") as f:
            json.dump(settings, f)
    
    
    def scroll_to_img_no(self):
        # Calculate the fraction needed to scroll to img_no
        fraction = (self.img_no) / (len(self.np_array))
    
        # To avoid scrolling out of bounds, we need to make some adjustments.
        # Let's say the canvas has a height of 600 pixels and each thumbnail has a height of 60 pixels.
        # Then in the last 10 images we don't want any more scrolling to prevent scrolling out of bounds.
    
        max_scrollable_img_no = len(self.np_array) - int(self.canvas_thumbnails.winfo_height() / 68)
    
        # If img_no exceeds the maximum scrollable index, it is set to that value.
        if self.img_no > max_scrollable_img_no:
            fraction = (max_scrollable_img_no) / (len(self.np_array))

        self.canvas_thumbnails.yview_moveto(fraction)

    
    
    def split_images_and_entropy(self, img_ent):
        images = [entry[0] for entry in img_ent]
        entropies = [entry[1] for entry in img_ent]
        return images, entropies
    
    
    def start_entropy_calculation(self):  # Call this when you start the entropy calculation
        # Create a new Toplevel window for progress bar
        self.progress_window = Toplevel(self.image_window)
        self.progress_window.title("Calculating Entropy...")

        # Add a label for information
        Label(self.progress_window, text="Please wait while calculating entropy...").pack(pady=10)

        # Create and pack the progress bar
        self.progress = ttk.Progressbar(self.progress_window, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(pady=20)


    def start_preprocess(self):
        #self.button_save.config(state=DISABLED)
        self.confirm_button.config(state=DISABLED)
        # Create a new Toplevel window for progress bar
        self.preprogress_window = Toplevel(self.image_window)
        self.preprogress_window.title("Calculating")

        # Add a label for information
        Label(self.preprogress_window, text="Please wait ...").pack(pady=10)

        # Create and pack the progress bar
        self.preprogress = ttk.Progressbar(self.preprogress_window, orient="horizontal", length=200, mode="determinate")
        self.preprogress.pack(pady=20)
    
    
    def thread_it(self, func, *args):
        """ Pack functions into threads """
        self.myThread = threading.Thread(target=func, args=args)
        self.myThread.daemon = True  # When the main thread exits, the sub-threads will follow and exit directly, regardless of whether the operation is completed or not.
        self.myThread.start()

    
    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.image_window.attributes("-fullscreen", self.is_fullscreen)
        return "break"

    
    def update_all_thumbnails(self):
        for idx in range(len(self.np_array)):
            if self.List_images[idx] is None:
                self.load_image_at_index(idx)
            thumbnail_img = self.List_thumbnail_images[idx]
            # Update button image
            self.frame_thumbnails.winfo_children()[idx].config(image=thumbnail_img)
            self.frame_thumbnails.winfo_children()[idx].image = thumbnail_img  # Keep reference
        pass
    
    
    def update_buttons(self):
        self.button_back.config(state=NORMAL if self.img_no > 0 else DISABLED)
        self.button_forward.config(state=NORMAL if self.img_no < len(self.List_images) - 1 else DISABLED)
    
    
    def update_confirm_button_state(self, event=None):
        if self.combo.get():  # If there's a value selected in the combobox
            #if self.combo_color.get():
                self.confirm_button.config(state=NORMAL)
        else:
            self.confirm_button.config(state=DISABLED)
    
    
    def update_image(self):
        self.canvas_image.itemconfig(self.image_on_canvas, image=self.List_photoimages[self.img_no])
        self.canvas_image.config(scrollregion=self.canvas_image.bbox(ALL))
        self.resize_image(self.zoom_percent)
        self.update_listbox()
    
    
    def update_images_from_array(self, np_array, index):
        # Convert numpy array to PIL Image
        img = Image.fromarray(np_array)

        # Update main image lists
        self.List_images[index] = img
        self.List_photoimages[index] = ImageTk.PhotoImage(img)

        # Create and update thumbnail
        thumbnail_size = (60, 60)  # You can adjust the size as needed
        thumbnail = img.copy()
        thumbnail.thumbnail(thumbnail_size)
        self.List_thumbnails[index] = thumbnail
        self.List_thumbnail_images[index] = ImageTk.PhotoImage(thumbnail)
    

    def update_listbox(self):
        for widget in self.frame_thumbnails.winfo_children():
            widget.config(relief=FLAT)
        self.frame_thumbnails.winfo_children()[self.img_no].config(relief=SOLID)
        self.canvas_thumbnails.yview_scroll(self.img_no - int(self.canvas_thumbnails.winfo_height() / 68), 'units')

    
    def update_preprogress(self, text, iteration, total, start_time=None):
        percent = (iteration / float(total)) * 100
        self.preprogress["value"] = percent
        self.preprogress.update()
    
    
    def update_progress(self, text, iteration, total, start_time=None):
        percent = (iteration / float(total)) * 100
        self.progress["value"] = percent
        self.progress.update()


    def update_status_bar(self):
        #file_path = os.path.join(self.directory, self.image_files[self.img_no])
        #image_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        #img = self.List_images[self.img_no]
        if self.entropies_calculated == 0:
            info_text = f"Try to calculate the entropy!"
        else:
            info_text = f"Entropy: {self.entropy[self.sorted_indices[self.img_no]]}"#"File: {self.image_files[self.img_no]}  |  Resolution: {img.width}x{img.height}  |  Size: {image_size:.2f} MB"
        self.status_bar.config(text=info_text)

    
def choose_directory():
    try:
        directory = filedialog.askdirectory()
        if directory:
            ImageViewer(directory)
    except FileNotFoundError:
        messagebox.showerror("Error", "The image file was not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def  choose_directory_class():
    try:
        directory = filedialog.askdirectory()
        if directory:
            ClassViewer(directory)
    except FileNotFoundError:
        messagebox.showerror("Error", "The image file was not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


root = Tk()
root.title("Main Menu")
choose_button = Button(root, text="Choose Directory for Entropy Caluculate", command=choose_directory)
#choose_button_class = Button(root, text="Choose Directory for Classification", command=choose_directory_class)
choose_button.grid(row=0, column=0)
#choose_button_class.grid(row=1, column=0)
root.mainloop()