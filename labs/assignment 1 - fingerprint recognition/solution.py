from PIL import Image
import os
import subprocess as sb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
        
        
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
    all_original_names = []
    all_original_scores = []
    
    for f in files:
        sub1 = f.split("_")[0]
        sub2 = f.split("_")[-2]

        fp = open(f"{path}/{f}")
        score = int(fp.read())
        fp.close()
        
        if sub1 == sub2:
            genuines.append(score)
            gen_names.append((sub1, sub2))
        else:
            impostors.append(score)
            imp_names.append((sub1, sub2))
        all_original_names.append(f)
        all_original_scores.append(score)
    return (impostors, imp_names), (genuines, gen_names), (all_original_scores, all_original_names)


def plot_bozorth3_comparisons(imp_scores, gen_scores, imp_names, gen_names):
    """
    Plot bozorth3 matching points for impostors and genuines on the same plot.
    
    Args:
    imp_scores (list): List of matching points for impostors.
    gen_scores (list): List of matching points for genuines.
    imp_names (list of tuples): List of tuples containing impostor subject pairs.
    gen_names (list of tuples): List of tuples containing genuine subject pairs.
    """
    
    # Convert tuples to string representations for x-axis labels
    imp_labels = ['_'.join(name) for name in imp_names]
    gen_labels = ['_'.join(name) for name in gen_names]
    
    # Plot impostor scores in red
    plt.plot([i for i in range(len(imp_scores))], imp_scores, 'ro-', label='Impostors')
    
    # Plot genuine scores in blue
    plt.plot([i for i in range(len(imp_labels), len(imp_labels) + len(gen_labels))], gen_scores, 'bo-', label='Genuines')
    

    # Add labels, title, and formatting
    plt.xlabel('Subject Pairs (sorted alphabetically)')
    plt.ylabel('Number of Matching Points')
    plt.title('Bozorth3 Matching Points: Impostors vs Genuines')
    

    # Combine labels for x-axis
    all_labels = imp_labels + gen_labels
    
    # Add a grid and legend
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def get_similarity_matrix(scores, names):
    names1 = ["_".join(name.split("_")[:2]) for name in names]
    names2 = ["_".join(name.split("_")[-2:]).split(".")[0] for name in names]
    
    ns1 = set(names1)
    ns2 = set(names2)
    
    # TODO sort
    
    n_rows = len(set(names1))
    
    matrix = []
    for s1 in ns1:
        s1_scores = []
        for s2 in ns2:
            score = 0
                
            for i, (n1, n2) in enumerate(zip(names1, names2)):
                if n1 == s1 and n2 == s2:
                    score += scores[i]             
            s1_scores.append(score)
        
        matrix.append(s1_scores)
    
    return matrix, ns1, ns2

def plot_similarity_matrix(similarity_matrix, ns1, ns2):
    """
    Plot the similarity matrix as a heatmap.
    
    Args:
    similarity_matrix (ndarray): A 2D numpy array representing the similarity matrix.
    subjects (list): List of unique subjects used for row and column labels.
    """
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Bozorth3 Matching Points')
    
    # Set the x and y ticks to the subject names
    plt.xticks(ticks=np.arange(len(ns1)), labels=ns1, rotation=90)
    plt.yticks(ticks=np.arange(len(ns2)), labels=ns1)
    
    plt.title('Similarity Matrix (Bozorth3 Matching Points)')
    plt.tight_layout()
    plt.show() 


if __name__ == "__main__":
    try:
        os.mkdir("data")
        os.mkdir("data/bmp")
        os.mkdir("data/png")
        os.mkdir("data/feats")
        os.mkdir("data/matches")
    except:
        pass
    
    subjects = get_subjects("data/bmp/")
    
    
    skip = True
    if not skip:
        # Convert
        files = os.listdir("./data/bmp/")
        for f in files:
            convert_to_grayscale(f"./data/bmp/{f}")

        # Get minutae
        files = os.listdir("./data/png")
        for f in files:
            get_features(f"./data/png/{f}")
            
        
        # Get number of matching points
        files = os.listdir("./data/feats")
        files = [f for f in files if ".xyt" in f]
        print(len(files), (len(files) - 1) ** 2)
        for i, f1 in enumerate(files):
            for f2 in files[i:]:
                if f1 == f2:
                    continue
            
                print(f1, f2)
                prefix = "./data/feats"
                
                run_bozorth3(f"{prefix}/{f1}", f"{prefix}/{f2}")

    # Plot scores
    impostors, genuine, original = get_scores("data/matches/")
    
    imp, imp_s = impostors
    gen, gen_s = genuine
    original_scores, original_names = original
    
    # plot_bozorth3_comparisons(imp, gen, imp_s, gen_s)
    
    
    subjects = sorted(list(set([int(i[0]) for i in imp_s] + [int(i[0]) for i in imp_s])))

    matrix, ns1, ns2 = get_similarity_matrix(original_scores, original_names)
    plot_similarity_matrix(matrix, ns1, ns2)
