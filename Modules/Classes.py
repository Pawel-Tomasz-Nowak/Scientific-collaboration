
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder, KBinsDiscretizer as KBD
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, r2_score, make_scorer
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator, clone
import prince

from string import ascii_uppercase

import pathlib as path
parent_cwd_dir = path.Path.cwd().parent #Find the parent directory for current working directory (CWD).

class RareClassAggregator(TransformerMixin, BaseEstimator):
    def __init__(self,  q:float = 0.1) -> None:
        assert isinstance(q, float) and 0< q <1, "Argument 'q' musi być liczbą zmiennoprzecinkową z przedziału (0;1)"
    
        self.q: float = q


    def validate_X(self, X:pd.DataFrame, cat_features:list[str]) -> None:
        "Sprawdza czy x spełnia warunki"
        assert isinstance(X, pd.DataFrame), "Argument 'X' musi być instancją klasy pd.DataFrame!"

        assert isinstance(self.cat_features, list)

        assert set(cat_features).issubset(self.cat_features), "Zbyt wiele zmiennych kategorycznych przekazałeś!"
        assert len(cat_features) > 0, "Przekazałeś pusty zbiór cech kategorycznych!"




    def fit(self, X:pd.DataFrame, y: None = None, cat_features:list[str] = [] ) -> "RareClassAggregator":
        """Wylicz próg częstowliwościowy dla zmiennej kategorycznej 'cat_feature'.
        Opis argumentów:
        ---------
        X:pd.DataFrame - Ramka danych typu pandas, która zawiera ceche 'cat_feature', której rzadkie cechy chcemy połączyć w jedną klase.
        y:pd.DataFrame - Argument nieuzywany, służy do zachowania spójności transformatotów.
        cat_features:list[str] - Lista nazw zmiennych kategorycznych.
        """

        #'y' musi być stale równy None. Nie jest on potrzebny dla tego estimatora.
        if y is not None:
            raise ValueError("Argument 'y' musi być zawsze ustawiony na wartość None!")
        

        self.cat_features = cat_features
        
        self.validate_X(X = X, cat_features = self.cat_features) #Sprawx, czy argument X spełnia podstawowe założenia.


        self.freq_thresholds:dict[str, float] = {} #Tabela przechowująca wszystkie progi częstotliwościowe dla każdej cechy kategorycznej
        self.cross_tabs: dict[str : pd.Series] = {} #Słownik służący do przechowywania tabeli krzyżowych każdej zmiennej kategorycznej.

        for cat_feature in self.cat_features:
            feature_crosstab:pd.Series = ( X[cat_feature]. #Znajdź tabelę krzyżową dla zmiennej kategorycznej 'cat_feature'
                                         value_counts(normalize = True, sort = True))

            self.freq_thresholds[cat_feature] = feature_crosstab.quantile(q = self.q) #Wylicz próg częstotliwościowy dla zmiennej 'cat_feature'

            self.cross_tabs[cat_feature] = feature_crosstab

        
        return self


    def transform(self, X:pd.DataFrame, cat_features:list[str]) -> pd.DataFrame:
        """Transformuje ramkę danych X, agregując rzadkie klasy zmiennej kategorycznej 'cat_feature' """
        self.validate_X(X = X, cat_features = cat_features) #Upewnij się, że X jest ramką danych oraz, że cat_features jest NIEPUSTYM podzbiorem zbioru cech X.


        for cat_feature in cat_features:
            feature_freqtable:pd.Series = self.cross_tabs[cat_feature]  #Znajdź tabelę krzyżową znormalizowaną dla zmiennej jakościowej cat_feature.

            #Stwórz agregowaną kolumnę cechy kategorycznej.
            aggregated_col:pd.Series =  X[cat_feature].apply(func = lambda v:   v if feature_freqtable[v] >= self.freq_thresholds[cat_feature] else "Other").astype(dtype = "string")
                                        
            
            X.loc[:, cat_feature]  = aggregated_col
     
        return X
    

    def get_feature_names_out(self,) -> None:
        pass



class MultiOutputLinearRegression(MultiOutputRegressor):
    def __init__(self,estimator, **kwargs) -> None:
        super().__init__(estimator = estimator,**kwargs)



    def fit(self, X:np.ndarray, y:np.ndarray):
        y:np.ndarray[int] =  OneHotEncoder(sparse_output = False).fit_transform(X = y.reshape(-1,1))


        super().fit(X = X, y = y)


    def predict(self, X) -> np.ndarray[int]:
        y_pred: np.ndarray = super().predict(X = X).argmax(axis = 1)
     
        return y_pred


class WrappedSequentialFeatureSelection(SFS):
    def __init__(self, cat_vars_idx:list[int], num_vars_idx:list[int],estimator,  **kwargs) -> None:
        """"
        Opis argumentów: \n
        ---------
        cat_vars:list[str] - Lista ciągów znaków, które identyfikują zmienne kategoryczne \n

        """""""""
        super().__init__(estimator = estimator,**kwargs)
        
        self.cat_vars_idx:list[int] = cat_vars_idx #Indeksy wszystkich zmiennych kategorycznych przyszłej ramki danych X.
        self.num_vars_idx:list[int] = num_vars_idx #Indeksy wszystkich zmiennych numerycznych przyszłej ramki danych X.

    

    def fit(self, X:np.ndarray, y:np.ndarray):
        super().fit(X = X, y =y)

        self.candidates_idx:list[int] = np.flatnonzero(a = self.support_) #Tablica indeksów zmiennych optymalnych.

        self.cat_candidates_idx:list[int] = np.intersect1d(ar1 = self.candidates_idx, ar2 = self.cat_vars_idx) #Tablica indeksów zmiennych kategorycznych optymalnych.
        self.noncat_candidates_idx:list[int] = np.setdiff1d(ar1 = self.candidates_idx, ar2 = self.cat_candidates_idx) #Tablica indeksów zmiennych numerycznych optymalnych.


      
        self.pre_predictive_transformer = ColumnTransformer(transformers = [("OHE", OneHotEncoder(sparse_output = False), self.cat_candidates_idx),
                                                                            ("Identity", FunctionTransformer(), self.noncat_candidates_idx)],
                                                                            remainder = "drop")

        self.X_train = self.pre_predictive_transformer.fit_transform(X = X)
        self.y_train = y.copy()


        return self


    
    def _get_best_new_feature_score(self, estimator, X:np.ndarray, y:np.ndarray, cv:int, current_mask):
        # Return the best new feature and its score to add to the current_mask,
        # i.e. return the best new feature and its score to add (resp. remove)
        # when doing forward selection (resp. backward selection).
        # Feature will be added if the current score and past score are greater
        # than tol when n_feature is auto,
      

        candidate_feature_indices = np.flatnonzero(~current_mask) #Indeksy zmiennych, które nie zostały jeszcze wybrane.
        scores = {}
        
    
        for feature_idx in candidate_feature_indices:
            candidate_mask:np.ndarray[bool] = current_mask.copy() #Tworzymy kopię tablicy maskowej cech, które już wybraliśmy.
            candidate_mask[feature_idx] = True #W miejsce feature_indx wstawiamy True. Udajemy, że wybraliśmy cechę feature_indx.

            candidates:np.ndarray[int] = np.flatnonzero(a = candidate_mask) #Znajdź bezwzględne indeksy zmiennych kandydatowych.

            cat_candidates_idx:np.ndarray[int] = np.intersect1d(ar1 = candidates, ar2 = self.cat_vars_idx) #Znajdż bezwzglęne indeksy zmiennych kategorycznych kandydatów.
            noCat_candidates_idx:np.ndarray[int] = np.setdiff1d(ar1 = candidates, ar2 = cat_candidates_idx)

  

            #Kodowator OHE, który przekształca na gorąco zmienne kategoryczne, a pozostałych zmiennych nie narusza.
            candidates_transformer = ColumnTransformer(transformers = [("OHE", OneHotEncoder(sparse_output = False), cat_candidates_idx),
                                                                       ("Identity", FunctionTransformer(), noCat_candidates_idx)],
                                                                         remainder = "drop")
            X_new = candidates_transformer.fit_transform(X = X)

        
            scores[feature_idx] = cross_val_score(
                estimator,
                X_new,
                y,
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            ).mean()


        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])

        
        return new_feature_idx, scores[new_feature_idx]
    

    def predict(self, X:np.ndarray):
        #X to jest oryginalna ramka danych, która zawiera cechy wszystkie, w tym cechy optymalne.
        self.estimator.fit(X = self.X_train, y = self.y_train) #X_train to jest  treningowa ramka danych z optymalnymi predyktorami


        #Kodowator OHE, który przekształca na gorąco zmienne kategoryczne, a pozostałych zmiennych nie narusza.
        candidates_transformer = ColumnTransformer(transformers = [("OHE", OneHotEncoder(sparse_output = False), self.cat_candidates_idx)
                                                                   ,("Identity", FunctionTransformer(), self.noncat_candidates_idx)], 
                                    remainder = "drop")



        X_new:np.ndarray = candidates_transformer.fit_transform(X = X)
   

        return self.estimator.predict(X = X_new)







class ModelComparator():
    def __init__(self,  Filename:str, target_var:str, dtypes:dict[str, str], Models:dict[str,  ], Models_hipparams:dict[str, dict],
                 n_splits:int = 5, train_size:float = 0.8, test_size:float = 0.2, bins:list[int] = [150, 250], show_plots:bool = True, quartile_discr:bool = False, quartile_classes:int = 0) -> None:
        """Konstruktor klasy, która trenuje modele. 
        Opis argumentów konstruktora:
        ---------
        Filename: str - Nazwa pliku, w której zawarta jest ramka danych typu pandas, która przechowuje zmienne objaśniające i zmienną objaśnianą. \n
        target_var:str - Nazwa zmiennej objaśnianej. \n

        dtypes:dict[str, "datatype"] - Typ danych każdej kolumny w pliku. \n

        Models:dict[str, "Estimator"] - Słownik, którego wartościami są instancje modeli, które chcemy wytrenować. Kluczami są umowne nazwy modeli. \n
        Models_hipparams:dict[str, dict] - Słownik, którego kluczami są umowne nazwy modeli, a którego wartościami są siatki parametrów danego modelu. \n

        n_splits:int - Total number of iteration of main loop. \n
        train_size:float - Percentage of training observations . \n
        test_size:float - Percentage of testing observations. \n

        bins:list[int]  - Edges of  the classes of the target variables.
        show_plots: bool - If True, show the plots of descriptive statistics.
        quartile_discr: bool - If True, we perform discretization by quartiles based on quartile_clasess.
        quartile_clases:int - (Only works if quartile_disc = True) - the number of quartiles to determine the edges of classes' intervals.

        ---------
        """
        #Reading the target dataset.
        self.Dataset:pd.DataFrame = self.read_dataframe(filename = Filename, sep = ';', dec_sep = ',') 


        self.features:list[str] = [feature for feature in self.Dataset.columns] #List of modified names of all features
        self.dtypes:dict[str, "dtype"] = {self.features[i]: dtypes[feature] for i, feature in enumerate(dtypes.keys())} #Don't forget to modify the names in the dtypes
            
    
        self.Dataset.columns = self.features
        self.Dataset.astype(dtype = self.dtypes)



        #Note: We're applying a following classification of variables:
        #A variable is (categorical only and only if its dtype is 'string') or (continous only and only if its dtype is 'Float') or (discrete only and only if its dtype is 'Int')
        #A variable is numerical only and only if it's continous or discrete.
        self.cat_features:list[str] = [feature for feature in self.features if self.dtypes[feature] == "string"] #Find the categorical variables.

        self.discr_features:list[str] = [feature for feature in self.features if pd.api.types.is_integer_dtype(self.dtypes[feature])] #Find the discrete variables.
        self.cont_features:list[str] = [feature for feature in self.features if pd.api.types.is_float_dtype(self.dtypes[feature])] #Find the continous variables.

        self.num_features:list[str] = self.discr_features + self.cont_features #The set of all numerical variables.


     
        self.target_var:str = target_var
        self.bins:list[int] = bins

        self.Cat_predictors:list[str] = []
        self.Num_predictors:list[str] = []
        self.PCA_predictors:list[str] = []

        self.Models:dict[str, "estimator"] = Models
        self.Models_hipparams:dict[str, dict] = Models_hipparams
        colors:list[str] = ['red','orange', 'yellow','cyan','blue']

        self.colors_for_models:list[str] = [ colors[i] for i in range(len(Models))]

        self.n_splits:int = n_splits
        self.train_size:float = train_size
        self.test_size:float = test_size

        self.show_plots:bool = show_plots

        self.quartile_discr:bool = quartile_discr
        self.quartile_classes:int = quartile_classes

        self.results_directory:path.Path = parent_cwd_dir / ("Classic_Labels" if quartile_discr is False else  "Quartile_Labels") #Find the destination path for result directory, depending on discretization-type.
        
        if not self.results_directory.exists():
            self.results_directory.mkdir()


        self.create_predictions_dataframe()
        self.model_names:list[str] = list(self.Models.keys())


        
    def read_dataframe(self, filename:str, 
                      sep:str =';', dec_sep:str =',') -> pd.DataFrame:
        """Odczytuje plik o nazwie filename i wprowadza dane z pliku do ramki danych."""
        Dataset:pd.DataFrame = pd.read_csv(filename,
                            sep=sep, decimal = dec_sep) #Wczytaj plik z danymi.


        return Dataset


    def plot_barplot(self, feature:str, display_xtick_labels:bool = False) -> None:
        """The function displays the barplot representing the relative frequencies of the levels of a given 'feature'.

        Parameters:
        feature:str - the categorical feature whose levels' frequencies will be displayed.
        display_xtick_labels: bool - If False (it's the default set), don't display xtickslabels. Other wise, display them.

        returns:
        None

        """

        barplot_figure = plt.figure() #Figure of the plot.
        axes = barplot_figure.add_subplot() #A specific axes for the plot.
     
        relat_freq:pd.DataFrame = self.Dataset[feature].value_counts(normalize = True, sort = False).reset_index()

        sns.barplot(data = relat_freq, x = feature, y = "proportion", ax = axes, 
                    edgecolor = "black")


        axes.set_ylabel(f"relative frequency of level") #Set the xlabel.
        axes.set_xlabel(f"{feature}'s levels") #Set the ylabel.


        if not feature.startswith("CO2 Emissions(g/km)"):
            axes.set_xticklabels([]) #Remove xticks.
        else:
            axes.set_xticklabels([str(i) for i in self.class_labels])
    

        axes.spines[["top", "right"]].set_alpha(0.5) #Set top and right spines' transparency to 0.5.


        axes.set_title(rf"Relatives frequencies of the {feature}'s levels") #Set the title of the graph.


        if display_xtick_labels == True:
            axes.set_xticklabels(labels = self.Dataset[feature].unique())

        barplots_directory =  self.results_directory/"BarPlots" #Find the path to directory containing all barplots. 
                                                        #If the directory doesn't exists, create one.

        if not barplots_directory.exists(): #Check if the barplots_directory doesn't exist.
            barplots_directory.mkdir() #If True, create one.
        
        barplot_filename: path.Path = barplots_directory/f"BarPlot_for_{feature}.png".replace("/", "~") #Creaet a UNIQUE  name for the barplot for a given feature.

        if  barplot_filename.exists():
            barplot_filename.unlink()


        barplot_figure.savefig(fname = barplot_filename)
        
            
    


    def compute_and_draw_correlation_matrix(self,) -> None:
        """The function computes and displays the correlation matrix of all continous features.
        
        Parameters:
        None

        returns:
        None

        """
        CorrMatrix:pd.DataFrame =  self.Dataset[self.cont_features].corr(method = "pearson")

        corr_mat_fig:plt.figure = plt.figure(dpi = 500)
        corr_mat_axes:plt.axes = corr_mat_fig.add_subplot()


        sns.heatmap(CorrMatrix, annot=True, cmap='magma', vmin=-1, vmax=1, ax = corr_mat_axes)

        corr_mat_axes.set_xticklabels("")
        corr_mat_axes.set_yticklabels("")


        corr_mat_axes.set_title(r"Correlation matrix for continous variables")
    
        corrmat_directory = self.results_directory/"CorrelationMatrix" #Find the path to directory containing correlation_matrix image. If the directory doesn't exists, create one.

        if not corrmat_directory.exists(): #Check if the corrmat_directory doesn't exist.
            corrmat_directory.mkdir() #If True, create one.

        corrmatrix_filename: path.Path = corrmat_directory/f"CorrelationMatrix.png" #Creaet a UNIQUE  name for the correlationmatrix path.

        if corrmatrix_filename.exists():
            corrmatrix_filename.unlink()

        corr_mat_fig.savefig(fname = corrmatrix_filename)
        

        
            

    def delete_quasiid_feature(self) -> None:
        """Funkcja usuwa quasi-identyfikator zmienną oraz jedną obserwację, która zawiera klasę, która występuje tylko raz."""

        self.Dataset.drop(columns = ["Model"], inplace = True)

        self.Dataset = self.Dataset.loc[self.Dataset["Fuel Type"]!="Other", :].copy()

    

    def plot_condidtioned_distribution(self, Condition:str) -> None:
        """Function draws conditioned KDEPlots for continous variables or barplots for discrete variables"""
        
        for num_feature in self.num_features:
            figure = plt.figure() 
            axes = figure.add_subplot()


            plot_type: str ="Barplot" if num_feature in self.discr_features else "KDE" #Determine the type of distribution.
            plot_title:str = rf"Conditional {plot_type} of {num_feature}"  #Set the title of the plot
           


            if num_feature in self.cont_features:
                sns.kdeplot(data = self.Dataset, x = num_feature, ax = axes, hue = Condition)
               
               
                if self.quartile_discr == False: #Set the verbal labels for the legend only if we're binning the target variable manually.
                    axes.legend(labels = ["wysoka", "średnia", "niska"], handles = axes.lines, title = "Klasy emisyjności") #Then set the labels for corresponding lines

            else:
                sns.histplot(data = self.Dataset, x = num_feature, ax = axes, hue = Condition, stat = "probability")

            axes.set_title(label = plot_title)
            axes.grid(True)
            
            KDEplots_directory = self.results_directory/"Conditioned_Distribution" #Find the path to directory containing boxplots image. If the directory doesn't exists, create one.
    
            if not KDEplots_directory.exists(): #Make sure the plot doesn't exist.
                KDEplots_directory.mkdir() #create one.

            KDEplot_filename: path.Path = KDEplots_directory/fr"{plot_title}.png".replace("/","~") #Creaet a UNIQUE  name for the KDE path for a given feature..
           

            if KDEplot_filename.exists():
                KDEplot_filename.unlink()

            
            figure.savefig(fname = KDEplot_filename)
            



    def conditioned_boxplot(self, Condition: str) -> None:
        """Draws conditioned boxplot for numerical variables using seaborn.boxplot function.
        """

        for num_feature in self.num_features:
            figure = plt.figure()
        
            axes = figure.add_subplot()
        
            sns.boxplot(self.Dataset, x = num_feature, hue = Condition, ax = axes)

            axes.set_title(rf"Boxplot for {num_feature} feature")

            boxplots_directory = self.results_directory/"Boxplots" #Find the path to directory containing boxplots image. If the directory doesn't exists, create one.

            if not boxplots_directory.exists(): #Check if the boxplot for the feature doesn't exist.
                boxplots_directory.mkdir() #If True, create one.

            boxplot_filename: path.Path = boxplots_directory/f"Conditioned boxplot for {num_feature}.png".replace("/", "~") #Create a name for the .png file.

            if boxplot_filename.exists():
                boxplot_filename.unlink()

            figure.savefig(fname = boxplot_filename)
            



    
    def plot_distribution(self) -> None:
        """Draws (violinplots for continous variables) or (boxplots for discrete variables)"""

        for feature in self.num_features:
            plot_type:str = "Boxplot" if feature in self.discr_features else "Violinplot" #Determine the type of the plot based on the variable's dtype.
            plot_title: str = f"{plot_type} for {feature}"

            figure = plt.figure()
            axes = figure.add_subplot()
         
            if feature in self.discr_features:
                sns.boxplot(data = self.Dataset,
                            x = feature, 
                            ax = axes)
            else:
                sns.violinplot(self.Dataset, 
                               x = feature, 
                               ax = axes, density_norm = "count")


            axes.set_title(plot_title)
            axes.legend([])
            axes.grid(True, alpha = 0.6)
            axes.spines[['top','right']].set_visible(False)


            violinplots_directory = self.results_directory/"DistributionPlots" #Find the path to directory containing violinplot image. If the directory doesn't exists, create one.

            if not violinplots_directory.exists(): #Check if the violinplot for the feature doesn't exist.
                violinplots_directory.mkdir() #If True, create one.

            violinplot_filename: path.Path = violinplots_directory/f"{plot_type} for {feature}.png".replace("/", "~") #Creaet a UNIQUE  name for the violinplot path for a given feature..

            if violinplot_filename.exists():
                violinplot_filename.unlink()

            figure.savefig(fname = violinplot_filename)

     


    def discretize(self) -> None:
        """Discretize the target variable with respect to given bins"""

        if self.quartile_discr == True:
            KBD_inst = KBD(n_bins = self.quartile_classes, encode = "ordinal", strategy  = "quantile")

            self.class_labels:list[int] =  [int(i) for i in range(self.quartile_classes-1)] #Find the list of class-labels.
            

            discretized_feature:np.ndarray = KBD_inst.fit_transform(X = pd.DataFrame(self.Dataset[self.target_var]))[:, 0]



        else:
            self.class_labels:list[int] =  [i for i in range(len(self.bins)-1)] #Find the list of class-labels.
        

            discretized_feature:pd.Series = pd.cut(x = self.Dataset[self.target_var],  #Finally, discretize the labels.
                                        bins = self.bins, 
                                        labels = self.class_labels)
            

        self.Dataset[self.target_var_discr] = discretized_feature #Add brand-new discretized feature to the dataset.


    def descriptive_statistics(self) -> None:
        """"This  methods shows some statistical properties of the features of the dataframe. In this section we also examine the discriminant-ability of the variables.
        In other words, we're manually looking for the most optimal candidates for predictors
        """

        #Nadaj zdyskretyzowanej zmiennej docelowej nazwę.
        self.target_var_discr = self.target_var +"_disc"

        # Agregacja rzadkich klas.
        RareClassAggregator_inst = RareClassAggregator(q = 0.15) #Zdefiniuj obiekt klasy RareClassAggregator, który będzie agregował rzadkie klasy każdej cechy.

        RareClassAggregator_inst.fit(X = self.Dataset, y = None,  #Znajdź odpowiednie parametry  estymatora.
                                     cat_features = self.cat_features)

        self.Dataset:pd.DataFrame = RareClassAggregator_inst.transform(X = self.Dataset, cat_features = self.cat_features) #Przekształć obecny zbiór danych.


        if self.show_plots:
            for finite_feature in self.cat_features + self.discr_features:
                self.plot_barplot(feature = finite_feature)


        #Delete the extreme-outsider record and redundant column.
        self.delete_quasiid_feature()


        #Discretize the response variable.
        self.discretize()


        
        #Narysuj wykresy charakteryzujące (relacje między zmiennymi) oraz (charakterystyki zmiennych).
  
        if self.show_plots is True:
            #Drawing a corelation matrix for continous variables.
            self.compute_and_draw_correlation_matrix()

            #RYSOWANIE WYKRESÓW SKRZYPCOWYCH DLA ZMIENNYCH NUMERYCZNYCH
            self.plot_distribution()

        
        
            self.plot_barplot(feature =  self.target_var_discr)  #Draw a barplot for discretized target feature.
                            
            
        
            self.plot_condidtioned_distribution(Condition = self.target_var_discr)  #Wykresy gęstości warunkowe
            self.conditioned_boxplot(Condition = self.target_var_discr) #Wykresy pudełkowe warunkowe

        
    

        #Ustal ostateczny zbiór predyktorów.
        self.predictors:list[str] = ['Make', "Vehicle Class",'Engine Size(L)','Cylinders','Transmission','Fuel Type',
                                    "Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)",  
                                        "Fuel Consumption Comb (L/100 km)","Fuel Consumption Comb (mpg)"]
       
      
        #Podziel zbiór predyktorów na zmienne numeryczne oraz zmienne kategoryczne odpowiednio.
        self.num_predictors_idx:list[int] = []
        self.cat_predictors_idx:list[int] = []

        
        self.PCA_predictors_idx:list[int] = []
       

        for idx, feature in enumerate(self.predictors):
            if feature in self.cat_features: #Sprawdzanie, czy zmienna jest zmienną kategoryczną.
                self.cat_predictors_idx.append(idx) #Dodaj indeks zmiennej kategorycznej do listy zmiennych kategorycznych.

            elif feature in self.num_features: #Sprawdzanie, czy zmienna jest typu numerycznego.
                self.num_predictors_idx.append(idx) #Dodaj indeks zmiennej numerycznej do listy zmiennych numerycznych.

                if feature.startswith("Fuel"): #Znajdź zmienną, które podlegają PCA.
                    self.PCA_predictors_idx.append(idx)
            
                    


    def create_predictions_dataframe(self,) -> None:
        """Creates a dataframe for storing the actual labels and labels predicted by each of the model. The column-system of the dataframe is four-level. The syntax for selecting a column is as follows:
        FactVsPrediction[('model_name', 'train_type', 'iter_idx, 'y_type')] \n
        'model_name' is a name of the model,  \n
        'train_type' is a type of training:  noFS_untuned, noFS_tuned, FS_untuned, FS_tuned, \n
        'iter_idx' is an index of the main iteration, \n
        'y_type' is a type of y labels: actual or predicted. \n

         """
        self.training_types: list[str] = ["noFS_untuned", "noFS_tuned",
                                         "FS_untuned","FS_tuned"]
                                    
        col_indeces = pd.MultiIndex.from_product( [list(self.Models.keys()),self.training_types, range(self.n_splits), ["True", "Pred"] ] #Stwórz  hierarchiczny system indeksów dla kolumn.
                                            ,names = ["model","train_type" ,"iter_idx", "y_type"]) #Nadaj poszczególnym poziomom wyjaśnialne i sensowne nazwy.
        
        row_indeces = np.arange(0, stop = np.ceil( self.Dataset.shape[0]* self.test_size))
        

        self.FactVsPrediction:pd.DataFrame =  pd.DataFrame(data = None,  columns = col_indeces,  index = row_indeces,
                                                           dtype = np.int16)




    def transform_predictors(self,  num_predictors_idx:list[int], PCA_predictors_idx:list[int] | None = None, cat_predictors_idx:list[int] | None = None, train_type: str = "noFS") ->  ColumnTransformer:
        """Transform the predictors. The exact  list of transformers is dependent on whether FS is included or not.
        If not (featsel = False), the following transformations are being applied: OrdinalEncoding, StandardScaler, PCA
        
        Parameters \n
        --------- \n
        num_predictors_idx : list[int] : list of indeces of numerical variables. \n
        PCA_predictors_idx : list[int] : list of indeces of PCA variables. \n
        cat_predictors_idx : list[int] : list of indeces of categorical variables.  \n
        featsel : str  = "noFS" : type of learning: noFS (noFeatureSelection), FS (FeatureSelection), FE (FeatureExtraction)

        ---------

        Returns: \n
        ColumnTransformer

        """
        doPCA: bool = train_type == "noFS" or train_type == "FS"

        if doPCA:
            noPCA_predictors_idx:list[int] = np.setdiff1d(ar1 = num_predictors_idx, ar2 = PCA_predictors_idx) #Find the NUMERICAL predictors which won't be PCA-transformed.
            noPCA_transformer = Pipeline(steps = [("Scaler", StandardScaler())]) #Define transformer for noPCA_predictors.
        

                                                
            PCA_transformer = Pipeline(steps = [("Scaler", StandardScaler()),   ("PCA", PCA(n_components =0.9))      ]) #Define transformer for PCA_predictors.
        

      
        if train_type == "noFS": #If  we're not including FeatureSelection, we're simply encoding categorical variables using OneHotEncoder.
            Predictors_transformer = ColumnTransformer(transformers = [("OHE", OneHotEncoder(sparse_output = False), cat_predictors_idx),
                                                                            ("Numerical", noPCA_transformer, noPCA_predictors_idx),
                                                                            ("PCA", PCA_transformer, PCA_predictors_idx)],
                                                                            remainder = "passthrough"
                                                                        )
            
        elif train_type == "FS": #However, if we are including FeatureSelection, first we encode categorical variables using OrdinalEncoder. 
                                    #OneHotEncoder for this type of training will be applied inside WrappedFeatureSelection class.
            Predictors_transformer = ColumnTransformer(transformers = [("ORD", OrdinalEncoder(), cat_predictors_idx),
                                                                            ("Numerical", noPCA_transformer, noPCA_predictors_idx),
                                                                            ("PCA", PCA_transformer, PCA_predictors_idx)],
                                                                            remainder = "passthrough"
                                                                      )
            Predictors_transformer.set_output(transform = "pandas") #Set the output type to output

        
        else:
            raise ValueError(f"Unsupported type of training {train_type} was passed ")


                
        return Predictors_transformer


    def train_with_FS(self, train_indx:np.ndarray, test_indx:np.ndarray,split_indx:int = 0) -> None:
        """Training  the machine learning models with SequentialFeatureSelection included. \n

        Parameters:
        --------- 
        train_indx : np.ndarray : A numpy ndarray of training indeces. \n
        test_indx : np.ndarray :  A numpy ndarray of testing indices . \n
        split_indx : int : Split indicator. \n
        --------

        Returns: 
        None
        """
        X_train:np.ndarray = self.X[train_indx, :] #Training set of possible predictors.
        X_test:np.ndarray = self.X[test_indx, :] #Testing set of possible predictors.

        y_train:np.ndarray = self.y[train_indx] #Training set of target variable.
        y_test:np.ndarray = self.y[test_indx] #Testing set of target variable
        

        predictors_transformer = self.transform_predictors(num_predictors_idx  = self.num_predictors_idx, #Define the predictors preprocessing transformer.
                                                             PCA_predictors_idx= self.PCA_predictors_idx,
                                                             cat_predictors_idx=self.cat_predictors_idx, 
                                                             train_type = "FS")
        predictors_transformer.fit(X = X_train) #Fit the transformer using TRAINING DATA to avoid data-leakage problem.

                                                        
        X_train:pd.DataFrame = predictors_transformer.transform(X = X_train) #Transform the training set X with the transformer.
        X_test:pd.DataFrame = predictors_transformer.transform(X = X_test) #Transform the testing set X with the transformer.

        #It's worth noting the dimensionality of the transformed training and testing set may change due to PCA, which is a reduction method.
        #Because of that, the relatives positions of variables may change. The following code finds the new indeces of both categorical and numerical variables.
        cat_vars_idx:list[int] = [] #An array of categorical variables' indces.
        num_vars_idx:list[int] = [] #An array of numerical variables' indeces.

        for i, var in enumerate(X_train.columns):
            if var.startswith("ORD"): #How can we recognize categorical variable? Well, it's been  transformed using OrdinalEncoder which adds prefix "ORD" to the variable' name.
                cat_vars_idx.append(i) #Add the index.
            
            else: #Otherwise it's a numerical predictor.
                num_vars_idx.append(i)



        for model_name in self.Models.keys():
            model  = clone(self.Models[model_name]) #The machine learning model we're training.

            model_paramgrid = self.Models_hipparams[model_name] #The parameters_space for the model.
            trans_model_paramgrid = {f"Model__{param}":model_paramgrid[param] for param in  model_paramgrid.keys()} #Adjust the names of the hyperparameters. Why? Because de facto we're tuning SeqFeatSel and we wanna tune the model.
          
                   


            SFS_inst = WrappedSequentialFeatureSelection(cat_vars_idx = cat_vars_idx, num_vars_idx = num_vars_idx,
                                              estimator = model, 
                                              n_features_to_select = "auto", tol = 0.01, n_jobs = -1, cv = 3, scoring = self.scoring_method)
            
            model_FS_tuned: Pipeline = Pipeline(steps = [("FeatSel", SFS_inst), ("Model", model)])
        
   
     
            GridSearch = GridSearchCV(estimator =model_FS_tuned , param_grid = trans_model_paramgrid,   #Define the GridSearch.
                                      n_jobs = -1, scoring = self.scoring_method, cv = 3, error_score = 0) 
            
            GridSearch.fit(X = X_train, y = y_train) #Train the GridSearch with training data.


            y_pred_tuned:np.ndarray = GridSearch.best_estimator_.predict(X = X_test) #Predict the labels using the FS_tuned model.



            SFS_inst.fit(X = X_train, y = y_train) #Fit the model FS_notuned.
            y_pred_untuned:np.ndarray = SFS_inst.predict(X = X_test) #Predict the labels with FS_notuned model.



            #Save the  both actual and predictes labels  for both FS_tuned and FS_untuned models.

            self.FactVsPrediction[(model_name, "FS_tuned", split_indx, "True")] = y_test
            self.FactVsPrediction[(model_name, "FS_tuned", split_indx, "Pred")] = y_pred_tuned

            self.FactVsPrediction[(model_name,"FS_untuned", split_indx, "True")] = y_test
            self.FactVsPrediction[(model_name,"FS_untuned",split_indx, "Pred")] = y_pred_untuned




    def train_without_FS(self, train_indx:np.ndarray, test_indx:np.ndarray, split_indx:int = 0) -> None:
        """Train the ML models without FeatureSelection.
        
        Parameters:
        --------- 
        train_indx : np.ndarray : A numpy ndarray of training indeces. \n
        test_indx : np.ndarray :  A numpy ndarray of testing indices . \n
        split_indx : int : Split indicator. \n
        --------

        Returns: 
        None

        """
        predictors_transformer:ColumnTransformer = self.transform_predictors( num_predictors_idx = self.num_predictors_idx, 
                                                                        PCA_predictors_idx = self.PCA_predictors_idx, 
                                                                        cat_predictors_idx = self.cat_predictors_idx,
                                                                        train_type = "noFS")
        
        X_train:np.ndarray = self.X[train_indx, :] #Treningowy zbiór predyktorów.
        X_test:np.ndarray = self.X[test_indx, :] #Testowy zbiór predyktorów.

        y_train: np.ndarray = self.y[train_indx]
        y_test: np.ndarray = self.y[test_indx]

        for model_name in self.Models.keys():
        
            model: 'estimator' = clone(self.Models[model_name]) #Instancja danego modelu.

            model_paramgrid: dict[str, dict] = self.Models_hipparams[model_name] #Siatka hiperparametrów modelu.


            trans_model = Pipeline(steps = [("Transformations", predictors_transformer), ("Classifier",model)]
                                       )
                 
            #Trenowanie modeli bez strojenia hiperparametrów.

            trans_model.fit(X = X_train, y = y_train)


            y_pred:np.ndarray = trans_model.predict(X = X_test)

      
            self.FactVsPrediction[(model_name, "noFS_untuned", split_indx, "True")] = y_test
            self.FactVsPrediction[(model_name, "noFS_untuned", split_indx, "Pred")] = y_pred
        


            #Trenowanie modeli ze strojeniem hiperparametrów.
            trans_model_paramgrid = {f"Classifier__{param}":model_paramgrid[param] for param in  model_paramgrid.keys()} #Dopasuj nazwy hiperparametrów do estymatora, 
                                                                                         # który nie jest bezpośrednio klasyfikatorem.

            GridSearch = GridSearchCV(estimator = trans_model, param_grid = trans_model_paramgrid, n_jobs = -1, scoring = self.scoring_method, cv = 3,error_score = 0)
          
            
            GridSearch.fit(X = X_train, y = y_train)
            y_pred:np.ndarray = GridSearch.predict(X = X_test)


            self.FactVsPrediction[(model_name, "noFS_tuned", split_indx, "True")] = y_test
            self.FactVsPrediction[(model_name, "noFS_tuned", split_indx, "Pred")] = y_pred



    def train_models(self) -> None:
        """"The  method  train the models with two versions: without FeatureSelection and with FeatureSelection. To split the dataset into training and testing subsets,
        a StratifiedShuffleSplit is defined so that the probabilities of targer's modalities are preserved.

        Parameters: \n
        ---------
        
        ---------
        Returns: \n
        None
        """
        #Define an instance of StratifiedShuffleSplit.
        SSS_inst: StratifiedShuffleSplit = StratifiedShuffleSplit(n_splits = self.n_splits, 
                                                                  test_size = self.test_size, 
                                                                  train_size = self.train_size)
        
        self.X:np.ndarray = self.Dataset[self.predictors].to_numpy() #The set X of predictors.
        self.y:np.ndarray = self.Dataset[self.target_var_discr].to_numpy()  #Discretized target variable.
        
        self.scoring_method : callable = make_scorer(f1_score, average = "weighted", zero_division =0)


        for split_indx, indx_tup in enumerate(SSS_inst.split(X = self.X, y = self.y)):
            train_indx, test_indx = indx_tup #Unpack the tuple of training index and testing index.
            print(f"Iteration no {split_indx}")

            y_train = self.y[train_indx]
            y_test = self.y[test_indx]


        
            self.train_without_FS( train_indx = train_indx, test_indx = test_indx, split_indx = split_indx) #Train the models without FeatureSelection.
            self.train_with_FS( train_indx = train_indx, test_indx = test_indx, split_indx = split_indx) #Train the model with FeatureSelection



    def compare_four_train_types(self, metrics_dataframe:pd.DataFrame, metrics_names:list[str]) -> None:
        "For each given model, compare the perfomance of its four versions using given metric"
        for model_name in self.model_names:
            for metric_name in metrics_names:
                boxplot_figure: plt.figure = plt.figure()
                boxplot_axes: plt.axes = boxplot_figure.add_subplot()


                slicer = pd.IndexSlice

                metrics_dataframe_melted:pd.DataFrame = metrics_dataframe.loc[:, slicer[model_name, :, metric_name]].melt(var_name = "train_type", value_name = "metric_value", col_level = 1)

                
        
                sns.boxplot(data = metrics_dataframe_melted, y = "metric_value", x = "train_type",ax = boxplot_axes,
                            palette = self.colors_for_models)


                boxplot_axes.legend(self.training_types)

                boxplot_axes.grid(True)
                boxplot_axes.spines[["top", "right"]].set_visible(False)

                boxplot_axes.set_xlabel("Training type")
                boxplot_axes.set_ylabel(f"{metric_name} values")

                boxplot_axes.set_title(f"Variability of {metric_name}  for {model_name}")

                boxplot_axes.set_xticks([i for i in range(len(self.training_types))])
                boxplot_axes.legend([])


                boxplot_metric_directory = self.results_directory/"Boxplot for metric and model" #Find the path to directory containing boxplots . If the directory doesn't exists, create one.

                if not boxplot_metric_directory.exists(): #Check if the boxplot  for the model and metric doesn't exist.
                    boxplot_metric_directory.mkdir() #If True, create one.

                boxplot_metric_filename: path.Path = boxplot_metric_directory/f"Boxplot for {metric_name} metric and {model_name} model.png" #Creaet a UNIQUE  name for boxplot for values of metric and model.

                if boxplot_metric_filename.exists():
                    boxplot_metric_filename.unlink()

                boxplot_figure.savefig(fname = boxplot_metric_filename)



    def plot_confussion_matrix(self) -> None:
        """The methods computes the confussion matrix which will be plotted as a heatmap"""
        dir_name:str = rf"Compacted confusion matrices"

        sel_training_types:list[str] = ["noFS_untuned", "FS_tuned"]

       
        for train_type in sel_training_types:
            for model_name in self.model_names:
                    graph_name: str = rf"Conf. Matrix, {model_name}" #The name of the axes an individual confusion matrix will be plotted on.
                    file_name = f"{train_type} - ConfMatrixOf{model_name}.png"

                    slicer = pd.IndexSlice
                    y_true:pd.Series = self.FactVsPrediction.loc[:, slicer[model_name, train_type, 0, "True"]] #Find the true label for the model and train type

                    y_pred:pd.Series = self.FactVsPrediction.loc[:, slicer[model_name, train_type, 0, "Pred"]]


            
                    figure = plt.figure()
                    conf_axes = figure.add_subplot()


                    ConfusionMatrixDisplay.from_predictions(y_true = y_true, y_pred = y_pred, normalize = "true", ax = conf_axes,
                                                            colorbar = False)
                    
                    conf_axes.set_xlabel(xlabel = "")
                    conf_axes.set_ylabel(ylabel = "")

                    conf_axes.set_title(graph_name)

                    conf_matrix_directory =   self.results_directory/dir_name #Create a  directory containing all confusion matrices for a given model.

                    if not conf_matrix_directory.exists():
                        conf_matrix_directory.mkdir()

                    conf_matrix_filename = conf_matrix_directory/file_name

                    if conf_matrix_filename.exists():
                        conf_matrix_filename.unlink()
                    
                    figure.savefig(fname = conf_matrix_filename)



    def plot_models_results_collectively(self, metrics_dataframe:pd.DataFrame, metrics_names:list[str]) -> None:
        for metric_name in metrics_names:
            for train_type in self.training_types:
                graph_name:str = rf"{train_type}: Models perfomance comparison using {metric_name}" #Create an informative and concise title for the plot.
                file_name:str = rf"{train_type} - Models perfomance comparison using {metric_name}"
                dir_name:str = rf"Models perfomance comparison"
                
                metric_figure:plt.Figure = plt.figure()
                metric_axes:plt.axes = metric_figure.add_subplot()

                
                y_values:pd.Series = metrics_dataframe.loc[:, (slice(None), train_type, metric_name)]

                sns.boxplot(data = y_values, ax = metric_axes)



                metric_axes.set_title(graph_name)
                metric_axes.set_xlabel("Model name")
                metric_axes.set_ylabel(f"{metric_name} values")
              
                metric_axes.grid(True, alpha = 0.7)


                boxplots_directory = self.results_directory/dir_name #Create a  directory containing all confusion matrices for a given model.

                if not boxplots_directory.exists():
                    boxplots_directory.mkdir()

                box_matrix_filename = boxplots_directory/file_name

                if box_matrix_filename.exists():
                    box_matrix_filename.unlink()
                
                metric_figure.savefig(fname = box_matrix_filename)



    def plot_median_values(self, metrics_dataframe:pd.DataFrame, metrics_names:list[str]) -> None:
        for metric_name in metrics_names:
            for train_type in self.training_types:
                graph_name: str = rf"{train_type}: Comparison of median values of {metric_name} of each model" #Create an informative and concise title for the plot.
                dir_name:str = rf"{train_type} Medianvalues of all models computed using"
                file_name = rf"{train_type} Medianvalues of all models computed using {metric_name}"

                medianmetric_figure = plt.figure() #Create a figure for dispalying the median values.
                medianmetric_axes = medianmetric_figure.add_subplot() #Create an axes associated with that figure.

                index_slicer = pd.IndexSlice #Define the instance of IndexSlice to make dataframes indexing easy.

                median_dataframe:pd.DataFrame = metrics_dataframe.loc[:, index_slicer[:, train_type, metric_name]].median(axis = 0).reset_index(level = [1,2], drop = True).reset_index()
                median_dataframe.columns =["Model", 'Median']

                min_value:float = median_dataframe['Median'].min() 

                sns.barplot(data = median_dataframe, x = "Model", y = "Median", palette = self.colors_for_models,
                            ax = medianmetric_axes, linewidth = 1.5, edgecolor = "black")
            

                medianmetric_axes.set_xlabel("Model", labelpad = 5)
                medianmetric_axes.set_ylabel(f"Median value", labelpad = 5)
                medianmetric_axes.set_title(graph_name)
                medianmetric_axes.set_ylim(0.99*min_value, 1)


                medvalues_directory = self.results_directory/dir_name #Create a  directory containing all confusion matrices for a given model.
                medvalues_filename = medvalues_directory/file_name

                if not medvalues_directory.exists():
                    medvalues_directory.mkdir()

                if medvalues_filename.exists():
                    medvalues_filename.unlink()
                
                medianmetric_figure.savefig(fname = medvalues_filename)



    def compute_perf_metric(self, metrics: dict[str, callable], metrics_names:list[str]) -> pd.DataFrame:
        """The function assess the perfomance of a given model, by a given training_type, by a given metric_name, by a given split_idx.

        Parameters: \n
        ---------
        metrics : dict[str, "metric"] : Dictionary holding the callable metrics. \n
        ---------

        Returns:
        pd.DataFrame
        """

        #The column-system of a metric_dataframe will be 3-leveled. The first level is model_name, the second is training_type, the third is metric_name.
        col_indeces = pd.MultiIndex.from_product(iterables =[self.model_names, self.training_types, metrics_names ]) 
        row_indeces: list[int] = list(range(self.n_splits))

        metrics_dataframe:pd.DataFrame = pd.DataFrame(data = None, index = row_indeces, columns = col_indeces)
        

        for model_name in self.model_names:
            for train_type in self.training_types:
                for metric_name in metrics_names:
                    for split_idx in row_indeces:
                        y_true:pd.Series = self.FactVsPrediction[(model_name, train_type, split_idx, "True")]
                        y_pred:pd.Series = self.FactVsPrediction[(model_name, train_type, split_idx, "Pred")]

                        if metric_name != "accuracy-score":

                            metric_value:float = metrics[metric_name](y_true = y_true, y_pred = y_pred, 
                                                                    average = "weighted", zero_division = 0)
                        else:
                            metric_value:float = metrics[metric_name](y_true = y_true, y_pred = y_pred, 
                                                                )
                            
                        metrics_dataframe.loc[split_idx,(model_name, train_type, metric_name)] = metric_value
                      
        return metrics_dataframe
    


    def compare_models(self) -> None:
        """Metoda wylicza, na podstawie przewidzianych przez modele etykiet, miary dokładności modelu, takie jak: accuracy_score, f1_score, precision_score, recall score.
        Następnie wyniki tych metryk przedstawia na wykresach."""
        #Zdefiniuj różne miary dokładności modeli.
        metrics:dict[str : "metric"] = {"accuracy-score":accuracy_score, "f1-score":f1_score}
    
        

        if self.quartile_discr:
            metrics:dict[str : "metric"] = {"f1-score":f1_score}
        else:
            metrics:dict[str : "metric"] = {"accuracy-score":accuracy_score, "f1-score":f1_score}

        metrics_names:list[str] = list(metrics.keys())
            

        metrics_dataframe:pd.DataFrame = self.compute_perf_metric(metrics  = metrics, metrics_names = metrics_names)


        self.plot_models_results_collectively(metrics_dataframe = metrics_dataframe, metrics_names = metrics_names)
        self.compare_four_train_types(metrics_dataframe = metrics_dataframe, metrics_names = metrics_names)
        self.plot_median_values(metrics_dataframe = metrics_dataframe, metrics_names = metrics_names)
        self.plot_confussion_matrix()