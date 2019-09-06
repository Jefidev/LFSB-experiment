import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from results_analysis.base_results_analysis import BaseResutsAnalysis


class EmbeddingAnalysis(BaseResutsAnalysis):
    def __init__(self):
        BaseResutsAnalysis.__init__(self)

    def run(self):
        self._load_embeddings()
        self._confusion_matrix()
        self._tsne()

    def _load_embeddings(self):
        logger.info("Loading {}".format(self.data))

        preds = "{}/{}".format(self.data, self.config["embedding_file"])
        labels = "{}/{}".format(self.data, self.config["label_file"])

        preds = np.load(preds, allow_pickle=True)
        labels = np.load(labels, allow_pickle=True)

        preds = pd.Series(preds.tolist())
        self.data_matrix = pd.DataFrame({"pred": preds, "label": labels})

    def _confusion_matrix(self):
        logger.info("creating confusion matrix")

        y_true = []
        y_pred = []

        for index, row in self.data_matrix.iterrows():
            true_label = row["label"]
            embed = np.array([row["pred"]])

            best_dist = 1000
            pred_label = ""

            for ifound, irow in self.data_matrix.iterrows():
                if ifound == index:
                    continue

                ie = np.array([irow["pred"]])
                dis = np.linalg.norm(embed[0] - ie[0])

                if dis < best_dist:
                    best_dist = dis
                    pred_label = irow["label"]

            y_true.append(true_label)
            y_pred.append(pred_label)

        confusion = confusion_matrix(y_true, y_pred)
        confusion = confusion / confusion.sum(axis=1)[:, np.newaxis]

        classes = unique_labels(y_true, y_pred)

        df_cm = pd.DataFrame(
            confusion, index=[i for i in classes], columns=[i for i in classes]
        )

        plt.figure(figsize=(7, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig("{}/{}".format(self.save_path, "confusion.png"))

    def _tsne(self):
        logger.info("Plotting TSNE")
        perplexity = [5, 15, 30, 35, 40, 45, 50, 60]

        for p in perplexity:
            r = self._reduce_dim(p)
            label = self.data_matrix["label"].values
            self._create_plot(r, label, p)

    def _reduce_dim(self, perplexity):
        preds = self.data_matrix["pred"].values.tolist()

        preds_reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(preds)
        return preds_reduced

    def _create_plot(self, preds_reduced, label, perplex):
        plot_df = pd.DataFrame(
            {"X": preds_reduced[:, 0], "Y": preds_reduced[:, 1], "label": label}
        )

        # plot_df["count"] = plot_df.apply(lambda row: len(plot_df[plot_df["label"] == row["label"]]), axis=1)
        # plot_df = plot_df[plot_df["count"] > 45]

        chart = (
            alt.Chart(plot_df)
            .mark_circle(size=60)
            .encode(x="X", y="Y", color="label")
            .interactive()
        )

        chart.save("{}/tsne_{}.png".format(self.save_path, perplex))
