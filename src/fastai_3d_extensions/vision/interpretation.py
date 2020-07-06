import fastai.vision as fvision
from fastai.vision import * #TODO
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from image import * 

class NiiClassificationInterpretation(fvision.Interpretation):
    "Interpretation methods for classification models."
    def __init__(self, learn:fvision.Learner, preds:fvision.Tensor, y_true:fvision.Tensor, losses:fvision.Tensor, ds_type:fvision.DatasetType=fvision.DatasetType.Valid):
        super(NiiClassificationInterpretation, self).__init__(learn,preds,y_true,losses,ds_type)
        self.pred_class = self.preds.argmax(dim=1)

    def confusion_matrix(self, slice_size:int=1):
        "Confusion matrix as an `np.ndarray`."
        x=torch.arange(0,self.data.c)
        if slice_size is None: cm = ((self.pred_class==x[:,None]) & (self.y_true==x[:,None,None])).sum(2)
        else:
            cm = torch.zeros(self.data.c, self.data.c, dtype=x.dtype)
            for i in range(0, self.y_true.shape[0], slice_size):
                cm_slice = ((self.pred_class[i:i+slice_size]==x[:,None])
                            & (self.y_true[i:i+slice_size]==x[:,None,None])).sum(2)
                torch.add(cm, cm_slice, out=cm)
        return to_np(cm)

    def plot_confusion_matrix(self, normalize:bool=False, title:str='Confusion matrix', cmap="Blues", slice_size:int=1, norm_dec:int=2, plot_txt:bool=True, return_fig:bool=None, **kwargs):
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
        cm = self.confusion_matrix(slice_size=slice_size)
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(self.data.c)
        plt.xticks(tick_marks, self.data.y.classes, rotation=90)
        plt.yticks(tick_marks, self.data.y.classes, rotation=0)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        if ifnone(return_fig, defaults.return_fig): return fig

    def plot_losses(self, k=3, most_confused=True, anatomical_plane:str=None, figsize=(10,10), return_fig:bool=False): 
        _,tl_idx = self.top_losses(k, largest=most_confused)
        classes = self.data.classes
        cols = math.ceil(math.sqrt(k))
        rows = math.ceil(k/cols)
        fig,axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle('index/prediction/actual/loss/probability', weight='bold', size=14)
        for i,idx in enumerate(tl_idx):
            im,cl = self.data.dl(self.ds_type).dataset[idx]
            cl = int(cl)
            show_image(im, ax=axes.flat[i], anatomical_plane=anatomical_plane, title=f'{int(idx)} / {classes[self.pred_class[idx]]}/{classes[cl]} / {self.losses[idx]:.2f} / {self.preds[idx][cl]:.2f}')
        for ax in axes.flatten(): ax.axis('off')
        
        if return_fig: return fig


    def most_confused(self, min_val:int=1, slice_size:int=1):
        "Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences."
        cm = self.confusion_matrix(slice_size=slice_size)
        np.fill_diagonal(cm, 0)
        res = [(self.data.classes[i],self.data.classes[j],cm[i,j])
                for i,j in zip(*np.where(cm>=min_val))]
        return sorted(res, key=itemgetter(2), reverse=True)
    
    def accuracy(self): 
        return accuracy_score(self.y_true, self.pred_class)