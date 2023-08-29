import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class ImageBrowser:

    def __init__(self, root):
        self.root = root
        self.root.title("Image Thumbnail Viewer")
        self.root.state('zoomed')

        # Create a button to select folder
        self.btn_select_folder = tk.Button(root, text="Select Folder", command=self.load_folder)
        self.btn_select_folder.pack(pady=10)

        # Create a frame to hold the canvas and scrollbar
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas to hold the thumbnails
        self.canvas = tk.Canvas(frame, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the canvas
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=scrollbar.set)

        # Update the scroll region on canvas configure
        self.canvas.bind('<Configure>', self.on_canvas_configure)

    def load_folder(self):
        folder_path = filedialog.askdirectory()

        if not folder_path:
            return

        # Clear previous images
        self.canvas.delete("all")

        # List all subdirectories
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        for idx, subdir in enumerate(subdirs):
            subfolder_path = os.path.join(folder_path, subdir)

            # Create a label for the subfolder name
            label = tk.Label(self.canvas, text=subdir, bg="gray")
            self.canvas.create_window((idx * 150 + 75, 20), window=label, anchor=tk.N)

            # Load images from the subfolder
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for jdx, img_file in enumerate(image_files):
                img_path = os.path.join(subfolder_path, img_file)
                with Image.open(img_path) as img:
                    img = img.resize((150, 150))
                    photo = ImageTk.PhotoImage(img)

                img_label = tk.Label(self.canvas, image=photo)
                img_label.image = photo
                self.canvas.create_window((idx * 150 + 75, jdx * 160 + 100), window=img_label, anchor=tk.N)

        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_canvas_configure(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageBrowser(root)
    root.mainloop()
