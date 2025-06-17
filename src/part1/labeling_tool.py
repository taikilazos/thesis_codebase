import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import nltk
from typing import List, Dict, Tuple
import re

class SentenceLabelingTool:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.current_sentence = ""
        self.current_sentence_id = 0
        self.sentences = []
        self.annotations = []
        
        # Labels for jargon types
        self.jargon_labels = [
            "Google Easy",
            "Google Hard",
            "Medical Name Entity",
            "Medical Abbreviation",
            "General Abbreviation",
            "General Complex Term",
            "Multi Sense Word"
        ]
        
        # Load and preprocess data
        self.load_data()
        
        # Setup GUI
        self.setup_gui()
    
    def load_data(self):
        """Load PLABA data and split into sentences"""
        # Load training data
        with open(os.path.join(self.data_dir, "train.json"), 'r') as f:
            train_data = json.load(f)
        
        # Process each document
        for doc_id in train_data:
            with open(os.path.join(self.data_dir, "abstracts", f"{doc_id}.src.txt"), 'r') as f:
                text = f.read().strip()
                # Split into sentences
                doc_sentences = nltk.sent_tokenize(text)
                self.sentences.extend([(doc_id, sent) for sent in doc_sentences])
    
    def setup_gui(self):
        """Setup the tkinter GUI"""
        self.root = tk.Tk()
        self.root.title("Sentence Labeling Tool")
        self.root.geometry("1000x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Sentence display
        self.text_widget = tk.Text(main_frame, height=5, width=80, wrap=tk.WORD)
        self.text_widget.grid(row=0, column=0, columnspan=2, pady=10)
        self.text_widget.bind("<ButtonRelease-1>", self.handle_selection)
        
        # Selected text display
        ttk.Label(main_frame, text="Selected Text:").grid(row=1, column=0, sticky=tk.W)
        self.selected_text_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.selected_text_var).grid(row=1, column=1, sticky=tk.W)
        
        # Jargon type selection
        ttk.Label(main_frame, text="Jargon Type:").grid(row=2, column=0, sticky=tk.W)
        self.jargon_type_var = tk.StringVar()
        jargon_type_combo = ttk.Combobox(main_frame, textvariable=self.jargon_type_var)
        jargon_type_combo['values'] = self.jargon_labels
        jargon_type_combo.grid(row=2, column=1, sticky=tk.W)
        
        # Sentence complexity
        ttk.Label(main_frame, text="Sentence Complexity:").grid(row=3, column=0, sticky=tk.W)
        self.complexity_var = tk.StringVar(value="complex")
        ttk.Radiobutton(main_frame, text="Complex", variable=self.complexity_var, 
                       value="complex").grid(row=3, column=1, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Simple", variable=self.complexity_var, 
                       value="simple").grid(row=3, column=1, sticky=tk.E)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Add Label", command=self.add_label).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Next Sentence", command=self.next_sentence).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        
        # Current annotations display
        ttk.Label(main_frame, text="Current Annotations:").grid(row=5, column=0, sticky=tk.W)
        self.annotations_text = tk.Text(main_frame, height=10, width=80)
        self.annotations_text.grid(row=6, column=0, columnspan=2)
        
        # Load first sentence
        self.load_current_sentence()
    
    def handle_selection(self, event):
        """Handle text selection"""
        try:
            selection = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.selected_text_var.set(selection)
        except tk.TclError:
            pass
    
    def add_label(self):
        """Add a label for the selected text"""
        try:
            start_idx = self.text_widget.index(tk.SEL_FIRST)
            end_idx = self.text_widget.index(tk.SEL_LAST)
            selection = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            
            if not selection or not self.jargon_type_var.get():
                messagebox.showwarning("Warning", "Please select text and jargon type")
                return
            
            # Add to current sentence annotations
            annotation = {
                "text": selection,
                "start": start_idx,
                "end": end_idx,
                "label": self.jargon_type_var.get()
            }
            
            # Update annotations display
            self.annotations_text.insert(tk.END, 
                f"{selection} -> {self.jargon_type_var.get()}\n")
            
        except tk.TclError:
            messagebox.showwarning("Warning", "Please select text to label")
    
    def load_current_sentence(self):
        """Load the current sentence into the GUI"""
        if self.current_sentence_id < len(self.sentences):
            doc_id, sentence = self.sentences[self.current_sentence_id]
            self.current_sentence = sentence
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, sentence)
            self.annotations_text.delete(1.0, tk.END)
    
    def next_sentence(self):
        """Save current sentence annotations and move to next"""
        # Save current sentence annotations
        if self.current_sentence_id < len(self.sentences):
            doc_id, sentence = self.sentences[self.current_sentence_id]
            
            # Get all annotations for current sentence
            annotations = []
            for line in self.annotations_text.get(1.0, tk.END).split('\n'):
                if '->' in line:
                    text, label = line.split('->')
                    text = text.strip()
                    label = label.strip()
                    if text and label:
                        annotations.append({
                            "text": text,
                            "label": label
                        })
            
            # Create MedReadMe format annotation
            tokens = self.current_sentence.split()
            annotation = {
                "tokens": tokens,
                "entities": annotations,
                "belongings": [self.complexity_var.get()],
                "split": "train"
            }
            
            self.annotations.append(annotation)
        
        # Move to next sentence
        self.current_sentence_id += 1
        if self.current_sentence_id >= len(self.sentences):
            messagebox.showinfo("Info", "All sentences have been labeled!")
            return
        
        self.load_current_sentence()
    
    def save_annotations(self):
        """Save annotations to file"""
        output_file = os.path.join(self.data_dir, "labeled_sentences.json")
        with open(output_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)
        messagebox.showinfo("Success", f"Annotations saved to {output_file}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    data_dir = "data/PLABA_2024-Task_1"
    tool = SentenceLabelingTool(data_dir)
    tool.run()

if __name__ == "__main__":
    main() 