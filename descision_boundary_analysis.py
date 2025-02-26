import yaml
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import pickle
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from main import get_model, get_loader

class DescisionBoundaryAnalyzer:
    def __init__(self, model, device='cuda', feature_name='assist_rd_assessment'):
        self.model = model
        self.device=device
        self.feature_name = feature_name # for filenaming purposes

    def get_embeddings(self, data_loader):
        """
        Extract embeddings and predictions for the provided samples
        """

        fname_list = []
        score_list = []
        embeddings_list = []
        confidence_list = []

        self.model.eval()

        for batch_x, utt_id in data_loader:
            
            batch_x = batch_x.to(self.device)
            
            with torch.no_grad():
                #  get embeddings from the final layer and the output
                batch_embeddings, batch_out = self.model(batch_x)
                
                sm = torch.nn.Softmax(dim=1)
                batch_confidences = sm(batch_out)
                batch_confidences = np.max(batch_confidences.data.cpu().numpy(), axis=1)
                
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

                print(batch_embeddings.shape)
                print(batch_score.shape)
            
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            confidence_list.extend(batch_confidences)
            embeddings_list.append(batch_embeddings.cpu().numpy())

        return fname_list, score_list, np.vstack(embeddings_list), confidence_list
    
        
    def compute_reduced_embeddings(self, data, method_name='umap', num_components=2, reduced_model_dir='./reduced_embeddings', apply_scaler=False, recompute_embeddings=False):
        """
        Compute the reduced embeddings and saves them in a directory
        """

        if not os.path.exists(reduced_model_dir):
            os.makedirs(reduced_model_dir)
    
        reduction_model_file = os.path.join(reduced_model_dir, self.feature_name + '_embed_' + str(num_components) + 'd.pkl')

        if apply_scaler:
            ss=StandardScaler().fit(data)
            data = ss.transform(data)

        # compute and save the embeddings
        if not os.path.exists(reduction_model_file) or recompute_embeddings:

            if method_name == 'umap':
                # compute umap embedding model
                reducer = umap.UMAP(n_components=num_components, random_state=42)

            else:
                reducer = TSNE(n_components=num_components, random_state=42)

            embeddings_mapper = reducer.fit(data)

            with open(reduction_model_file,'wb') as f:
                pickle.dump(reducer, f)

        else:
            with open(reduction_model_file, 'rb') as f:
                reducer = pickle.load(f)

        reduced_embeddings = reducer.transform(data)

        return reduced_embeddings

    
    def visualize_embeddings(self, reduced_embed_df, output_directory='./visualizations', regenerate_fig=False):
        """
        Visualize the reduced embeddings.

        Parameters:
            data: dataframe containing filenames, 2d embedddings of shape (n,2), labels, and scores
            feature_type (str): The feature type being visualized (for file naming).
            output_directory (str): Directory to save the plots.
        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        fig_output_path = os.path.join(output_directory, f'{self.feature_name}_visualiation.png')

        if not os.path.exists(fig_output_path) or regenerate_fig:

            # # Create a DataFrame for visualization
            # visualization_df = pd.DataFrame({
            #     'X1': reduced_embed[:, 0],
            #     'X2': reduced_embed[:, 1],
            #     'label': labels
            # })

            # Define a color map: "Original" is green, others in warm colors
            # unique_labels = visualization_df['label'].unique()
            # warm_colors = px.colors.sequential.solar[1:]  # Modify this list to your preferred warm color scheme
            # color_map = {label: "green" if label == 'bonafide' else warm_colors[i % len(warm_colors)]
            #             for i, label in enumerate(unique_labels)}

            # Plot using Plotly
            # warm_colors = px.colors.sequential.solar[1:]  # Modify this list to your preferred warm color scheme
            # symbols = ['circle-open', 'x']
            # color_map = {'bonafide': alphebet_colors[19], 'happy': alphebet_colors[6], 'sad': alphebet_colors[21], 'angry': alphebet_colors[17], '<unk>': alphebet_colors[4]}
            fig = px.scatter(
                reduced_embed_df, 
                x='X1', 
                y='X2', 
                color='Key_Result',
                # color='KEY', 
                # symbol = 'Result', 
                # symbol_sequence = symbols,
                labels={'label': 'KEY'},
            )

            fig.update_layout(title_text="Visualization of {}".format(self.feature_name), title_x=0.5)

            # Save the plot as HTML
            # os.makedirs(output_directory, exist_ok=True)
            # html_output_path = os.path.join(output_directory, f'{feature_type}_umap.html')
            # fig.write_html(html_output_path)
            # print(f"Plot saved to {html_output_path}")

            # Save the plot as PNG
            fig.write_image(fig_output_path)
            print(f"Plot saved to {fig_output_path}")

        else:

            print(f"{fig_output_path} already exists!!!")

        
    def analyze_confidence_distribution(self, eval_df, output_directory='./visualizations'):
        """
        Analyze and visualize confidence score distributions
        """

        confidences_correct = eval_df.loc[eval_df['Result']=='Correct', ['confidences']]
        confidences_incorrect = eval_df.loc[eval_df['Result']=='Incorrect', ['confidences']]

        plt.hist(confidences_correct, density=True, bins=20, label='Correct Predictions')
        plt.hist(confidences_incorrect, density=True, bins=20, label='Incorrect Predictions')

        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.legend()

        out_distribution_file = os.path.join(output_directory, f'{self.feature_name}_distribution_plot.png')
        plt.savefig(out_distribution_file)

    
    def find_boundary_indices(self, eval_df, n_samples=10):
        """
        find samples closeer to the boundary of the classifier
        """

        confidences = eval_df['confidences'].to_numpy()

        eval_df['uncertainty'] = np.abs(confidences - 0.5)

        eval_df_sorted = eval_df.sort_values(by='uncertainty')

        # print(eval_df_sorted)

        boundary_df = eval_df_sorted.head(n_samples)

        print(boundary_df)

        boundary_df.to_csv('./boundary_files.csv')

        return boundary_df
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RD Audio Spoof Detection Assessment")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    
    parser.add_argument("--compute_embeddings",
                        type=bool,
                        default=False,
                        help="set it true if needs to compute the embeddings again")

    
    args = parser.parse_args()
    
    # load experiment configurations
    with open(args.config, "r") as y_file:
        config = yaml.safe_load(y_file)

    model_config = config["model_config"]
    data_config = config["data_config"]  

    # define the database paths and protocol files
    track = data_config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    prefix_2019 = "ASVspoof2019_{}".format(track)  
    
    eval_pf = os.path.join(data_config["database_path"], "{}_eval_subset.tsv".format(track))
    eval_db_path = os.path.join(data_config["database_path"], "{}_eval_subset".format(prefix_2019))

    device = "cuda:1"
    feature_name = 'assist_rd_assessment'

    model = get_model(model_config, device)

    model.load_state_dict(torch.load(config["model_path"], map_location=device))
     
    print("Model loaded : {}".format(config["model_path"]))
    print("Start evaluation...")

    # create the dataloader
    eval_loader = get_loader(eval_db_path, eval_pf, args.seed, config=config, data_type='eval')

    analyzer = DescisionBoundaryAnalyzer(model, device, feature_name=feature_name)


    fname_ls, score_ls, eval_embeddings, confidences = analyzer.get_embeddings(eval_loader)

    print(len(fname_ls))
    print(len(score_ls))
    print(eval_embeddings.shape)

    
    reduced_embeddings = analyzer.compute_reduced_embeddings(eval_embeddings, apply_scaler=True, recompute_embeddings=True)
    print(reduced_embeddings.shape)

    preds_ls = ['bonafide' if scr>0 else 'spoof' for scr in score_ls]

    # create a dataframe with filenames and reduced embeddings
    red_embed_df = pd.DataFrame({
        'AUDIO_FILE_NAME': fname_ls,
        'X1': reduced_embeddings[:, 0],
        'X2': reduced_embeddings[:, 1],
        'Score': score_ls,
        'predictions': preds_ls,
        'confidences': confidences
        })

    eval_df = pd.read_csv(eval_pf, sep='\t') 

    reduced_embed_eval_df = pd.merge(red_embed_df, eval_df, on='AUDIO_FILE_NAME')

    reduced_embed_eval_df['Result'] = reduced_embed_eval_df['KEY'] == reduced_embed_eval_df['predictions']

    reduced_embed_eval_df['Result'] = reduced_embed_eval_df['Result'].map({True: 'Correct', False: 'Incorrect'})

    reduced_embed_eval_df["Key_Result"] = reduced_embed_eval_df["KEY"] + ' ' + reduced_embed_eval_df["Result"]

    print(reduced_embed_eval_df)

    reduced_embed_eval_df.to_csv("reduced_embed_eval.csv")

    # visualize the embeddings
    analyzer.visualize_embeddings(reduced_embed_df=reduced_embed_eval_df, regenerate_fig=True)
    
    # plot correct and incorrect distributions
    analyzer.analyze_confidence_distribution(eval_df=reduced_embed_eval_df)

    # find bounadry cases
    boundary_df = analyzer.find_boundary_indices(eval_df=reduced_embed_eval_df, n_samples=10)


            

            
        