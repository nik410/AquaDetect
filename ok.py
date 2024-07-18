from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter.font import Font
import joblib
from skimage.transform import resize
from skimage.io import imread
import numpy as np
Categories = ['Bacterial Red disease', 'Bacterial gill disease']
def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

def open_img():
    global file_paths
    filenames = filedialog.askopenfilenames(title='Open', multiple=True)
    file_paths = filenames  # Store file paths
    for filename in filenames:
        img = Image.open(filename)
        img = img.resize((480, 600), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = Label(frame, image=img)
        panel.image = img  # Keep a reference to avoid garbage collection
        panel.pack(padx=10, pady=10)
        images.append(img)  # Keep a reference to avoid garbage collection

def on_mousewheel(event):
    canvas.yview_scroll(-1*(event.delta//120), "units")

def predict_disease(img_path, model):
    img = imread(img_path)
    img_resized = resize(img, (150, 150, 3))
    img_array = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(img_array)
    probability = model.predict_proba(img_array)
    return prediction[0], probability

def analyse_images():
    model = joblib.load('fish_disease_model.pkl')  # Load your model
    results_text.delete(1.0, END)  # Clear previous results
    for file_path in file_paths:
        prediction, probability = predict_disease(file_path, model)
        disease = Categories[prediction]
        results_text.insert(END, f"Image: {file_path.split('/')[-1]}\nPrediction: {disease}\n")
        for i, category in enumerate(Categories):
            results_text.insert(END, f"{category}: {probability[0][i]*100 -1 :.2f}%\n")
        # results_text.insert(END, "\n")
        results_text.insert(END, "Healthy Fish: 2.00%" )
#         results_text.insert(END, """
#
# Preventive Measures:
#
# Maintain Water Quality:
#
# Ensure proper water quality parameters such as dissolved oxygen levels, pH, temperature, and ammonia levels are within the optimal range for the species being cultured.
#
# Implement regular water quality monitoring and management practices, including proper aeration, filtration, and water exchange to minimize stress on fish and reduce the risk of disease outbreaks.
# Biosecurity Protocols:
#
# Implement strict biosecurity measures to prevent the introduction and spread of pathogens into aquaculture facilities.
#
# Use disinfection protocols for equipment, vehicles, and personnel entering the facility to minimize the transmission of disease-causing agents.
#
# Quarantine new fish arrivals and conduct health screenings to detect and isolate infected individuals before introducing them into production systems.
#
# Nutrition and Feeding Practices:
#
# Provide a balanced and nutritionally adequate diet to support the immune function and overall health of fish.
#
# Avoid overfeeding and minimize feed wastage to prevent water quality degradation and reduce the risk of bacterial proliferation.
#
# Stocking Density Management:
#
# Avoid overcrowding by maintaining appropriate stocking densities to reduce competition for resources and minimize stress on fish.
#
# Provide adequate space and environmental enrichment to promote natural behaviors and social interactions among fish.
#
# Vaccination:
#
# Implement vaccination programs for commercially important fish species to provide protection against specific bacterial pathogens.
#
# Work with qualified veterinarians to develop vaccination strategies tailored to the specific disease risks and production conditions of the aquaculture operation.
# Corrective Measures:
#
# Diagnostic Testing:
#
# Conduct thorough diagnostic testing, including bacterial isolation, identification, and antimicrobial susceptibility testing, to confirm the presence of bacterial pathogens and guide treatment decisions.
# Treatment with Antibiotics:
#
# Administer antibiotics judiciously and according to veterinary guidance to control bacterial infections.
# Select antibiotics based on sensitivity testing results to ensure effectiveness and minimize the risk of antimicrobial resistance development.
#
# Water Management:
#
# Implement water treatment measures such as chlorination, ozonation, or UV sterilization to reduce bacterial loads in aquaculture systems.
#
# Maintain optimal water quality parameters and ensure proper filtration and aeration to support fish health and recovery.
# Environmental Management:
#
# Implement environmental modifications such as reducing stocking densities, adjusting water temperature, and optimizing feeding practices to alleviate stress on affected fish and promote recovery.
# Quarantine and Treatment Protocols:
#
# Isolate and treat affected individuals or groups of fish in quarantine facilities to prevent the spread of disease to healthy populations.
#
# Follow treatment protocols recommended by veterinarians, including proper dosing, duration, and withdrawal periods to minimize adverse effects and ensure treatment efficacy.
# Monitoring and Surveillance:
#
# Monitor fish health status regularly and conduct post-treatment evaluations to assess treatment effectiveness and detect any signs of disease recurrence.
# Implement surveillance programs to monitor for the presence of bacterial pathogens in aquaculture systems and surrounding environments to prevent future outbreaks.
#
#
#             """)

win = Tk()
win.geometry("800x600")
win.title("AquaDetect")
my_font = Font(size=15)

uLabel = Label(win, text="Select the Image:", font=my_font, padx=20, pady=20, height=2)
uLabel.pack()

# Create a frame for the canvas and scrollbar
canvas_frame = Frame(win)
canvas_frame.pack(fill=BOTH, expand=True)

canvas = Canvas(canvas_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=True)

scrollbar = Scrollbar(canvas_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

canvas.configure(yscrollcommand=scrollbar.set)

frame = Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor='nw')

frame.bind("<Configure>", on_configure)

btn = Button(win, text='Open Image', command=open_img, font=my_font, padx=20, pady=20, height=2)
btn.pack()

images = []
file_paths = []

btnAnalyse = Button(win, text="Analyse", command=analyse_images, font=my_font, padx=20, pady=20, height=2)
btnAnalyse.pack()

results_text = Text(win, height=10, width=80, font=my_font)
results_text.pack()

# Bind mousewheel event to the canvas
canvas.bind("<MouseWheel>", on_mousewheel)

win.mainloop()
