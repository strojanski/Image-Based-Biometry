from PIL import Image
import os
import subprocess as sb
import matplotlib.pyplot as plt
        
        
subjects = []

def convert_to_grayscale(input_path):
    output_path = input_path.replace(".bmp", ".png").replace("bmp", "png")
    with Image.open(input_path) as img:
        # Convert the image to 8-bit grayscale ('L' mode in Pillow)
        grayscale_img = img.convert('L')
        grayscale_img.save(output_path, 'PNG')
    return output_path
       

def get_features(f):
    output_file = f.replace(".png", "")
    output_file = f.replace("png", "feats")
    command = ["mindtct", f, output_file]
    
    try:
        result = sb.run(command, check=True)
    except sb.CalledProcessError as e:
        print("Error:", e)

def run_bozorth3(inp1, inp2):
    """Run bozorth3 and capture output"""
    try:
        # Run the bozorth3 command and capture the output
        result = sb.run(
            ["bozorth3", inp1, inp2], 
            stdout=sb.PIPE, 
            stderr=sb.PIPE, 
            check=True
        )
        
        # Decode the standard output (stdout)
        output = result.stdout.decode('utf-8')
        inp1 = inp1.replace(".xyt", "").replace(".feats", "")
        inp2 = inp2.replace(".xyt", "").replace(".feats", "")
        with open(f"data/matches/{inp1.split('/')[-1]}_vs_{inp2.split('/')[-1]}.txt", "w") as f:
            f.write(output)
        
        return output
    except sb.CalledProcessError as e:
        # Capture and print the error message if command fails
        error_output = e.stderr.decode('utf-8')
        print("Bozorth3 Error:\n", error_output)
        return None

def get_subjects(path):
    subs = []
    for f in os.listdir(path):
        f = f.split(".")[0]
        subs.append(f)
        
    return subs


def get_scores(path):
    files = os.listdir(path)
    
    impostors, genuines = [], []
    imp_names, gen_names = [], []
    
    for f in files:
        sub1 = f.split("_")[0]
        sub2 = f.split("_")[-2]

        fp = open(f"{path}/{f}")
        score = fp.read()
        fp.close()
        
        if sub1 == sub2:
            genuines.append(score)
            gen_names.append((sub1, sub2))
        else:
            impostors.append(score)
            imp_names.append((sub1, sub2))
        
    return (impostors, imp_names), (genuines, gen_names)

import matplotlib.pyplot as plt

def plot_bozorth3_comparisons(subject_pairs, matching_points, labels):
    """
    Plot bozorth3 matching points for impostors and genuines on the same plot.
    
    Args:
    subject_pairs (list): List of subject pair identifiers (e.g., '1_1', '1_2').
    matching_points (list): List of bozorth3 matching points corresponding to each subject pair.
    labels (list): List of labels ('genuine' or 'impostor') corresponding to each subject pair.
    """
    
    # Separate impostors and genuines
    impostor_pairs = [pair for pair, label in zip(subject_pairs, labels) if label == 'impostor']
    impostor_points = [point for point, label in zip(matching_points, labels) if label == 'impostor']
    
    genuine_pairs = [pair for pair, label in zip(subject_pairs, labels) if label == 'genuine']
    genuine_points = [point for point, label in zip(matching_points, labels) if label == 'genuine']
    
    # Sort impostor and genuine pairs alphabetically
    sorted_impostor_pairs, sorted_impostor_points = zip(*sorted(zip(impostor_pairs, impostor_points)))
    sorted_genuine_pairs, sorted_genuine_points = zip(*sorted(zip(genuine_pairs, genuine_points)))
    
    # Plot impostor scores in red
    plt.plot(sorted_impostor_pairs, sorted_impostor_points, 'ro-', label='Impostors')
    
    # Plot genuine scores in blue
    plt.plot(sorted_genuine_pairs, sorted_genuine_points, 'bo-', label='Genuines')

    # Add labels, title, and formatting
    plt.xlabel('Subject Pair (sorted alphabetically)')
    plt.ylabel('Number of Matching Points (Bozorth3 Score)')
    plt.title('Bozorth3 Matching Points: Impostors vs Genuines')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add a grid and legend
    plt.grid(True)
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example data:
subject_pairs = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2']  # Replace with your subject pairs
matching_points = [120, 150, 200, 180, 110, 130]  # Replace with corresponding bozorth3 scores
labels = ['genuine', 'impostor', 'impostor', 'genuine', 'impostor', 'genuine']  # Labels for impostor or genuine

# Plot the data
plot_bozorth3_comparisons(subject_pairs, matching_points, labels)



if __name__ == "__main__":
    try:
        os.mkdir("data")
        os.mkdir("data/bmp")
        os.mkdir("data/png")
        os.mkdir("data/feats")
        os.mkdir("data/matches")
    except:
        pass
    
    sub