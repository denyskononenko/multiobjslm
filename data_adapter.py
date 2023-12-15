import numpy as np
import pandas as pd

class DatasetPrint3D:
    """ 
    Adapter for dataset for 3D printing  with 
    Zr52.5Cu17.9Ni14.6Al10Ti5 alloy.

    Dataset includes measuremets of printed sample 
    density and amorphicity for three parametrs of 
    which regulates printing process:
    i)   Power [W]
    ii)  Hatching distance [\mu m]
    iii) Scanning velocity [m / s]
    
    Measurements are performend on the homogeneous 
    mesh of parameters. For each variant of params.
    three measurements are made.
    """
    
    def __init__(self, data_file: str, label: str):
        self.data_file = data_file
        self.label = label
    
    @property
    def data_pd_init(self):
        """
        Dataset with all three measurements for given parameters 
        as pandas DataFrame
        """
        return pd.read_excel(self.data_file, header=2)
    
    @property
    def data_pd_aver(self):
        """
        Dataset with averaged measuremets for given parameters 
        as pandas DataFrame
        """
        # dataset as numpy array
        dfn = self.data_pd_init.to_numpy()
        # columns labels 
        cols_labels =  self.data_pd_init.columns 
        # dictionary for averaged values for each 
        cols_aver = {label: [] for label in cols_labels}
        # indices of columns to average every three rows
        cols_to_aver_id = [4, 5, 6, 7] 

        # average each 3 measurements 
        # iterate over all rows
        for i in range(0, dfn.shape[0], 3):
            # iterate over all columns
            for j in range(cols_labels.shape[0]):
                temp_col_label = cols_labels[j]
                temp_col_label_std = f"{temp_col_label}_std"

                if j not in cols_to_aver_id:
                    temp_col_abbr = dfn[i:i+3, j]
                    cols_aver[temp_col_label].append(temp_col_abbr[0])
                else:
                    temp_col_mean = np.mean(dfn[i:i+3, j])
                    temp_col_std  = np.std(dfn[i:i+3, j])
                    cols_aver[temp_col_label].append(temp_col_mean)
                    
                    if temp_col_label_std not in cols_aver:
                        cols_aver[temp_col_label_std] = []
                    cols_aver[temp_col_label_std].append(temp_col_std)
        
        # add columns with Power Density and Entalpy 
        cols_aver['Energy density [MJ / m^2]'] = [self.calc_power_density(p, h, v) for p, h, v in zip(cols_aver['Power [W]'], cols_aver['hatching distance [µm]'], cols_aver['scanning velocity [m/s]'])]
        cols_aver['Enthalpy'] = [self.calc_enthalpy(p, v) for p, v in zip(cols_aver['Power [W]'], cols_aver['scanning velocity [m/s]'])]
        cols_aver['Mod Enthalpy 1'] = [self.calc_mod_enthalpy_1(p, h, v) for p, h, v in zip(cols_aver['Power [W]'], cols_aver['hatching distance [µm]'], cols_aver['scanning velocity [m/s]'])]
        cols_aver['Mod Enthalpy 2'] = [self.calc_mod_enthalpy_2(p, h, v) for p, h, v in zip(cols_aver['Power [W]'], cols_aver['hatching distance [µm]'], cols_aver['scanning velocity [m/s]'])]
        cols_aver['label'] = [self.label for _ in cols_aver['Power [W]']]

        aver_df = pd.DataFrame(data=cols_aver) 
        return aver_df
    
    @property
    def data_pd_pca(self):
        """ """
        dataset = self.data_pd_aver[[
            'Power [W]', 
            'hatching distance [µm]', 
            'scanning velocity [m/s]',
            'Enthalpy',
            'Energy density [MJ / m^2]']]
        
        return PCA(dataset, standardize=True).factors

    @property
    def X_Density(self):
        """Averaged measurements of the density."""
        dfn = self.data_pd_aver.values
        # power hatching_distance scanning_velocity
        X = dfn[:, [1,2,3]]
        # normalize density
        Y = dfn[:, 7] # / 100
        # standart deviation of density
        Y_std = dfn[:, 11]
        return X, Y, Y_std

    @property
    def X_Amorphicity(self):
        """Averaged measurements of the amorphisity."""
        dfn = self.data_pd_aver.values
        # power hatching_distance scanning_velocity
        X = dfn[:, [1,2,3]]
        # amorphisity
        Y = dfn[:, 5] # / 100
        # standart deviation of amorphisity
        Y_std = dfn[:, 9]
        return X, Y, Y_std

    @property
    def X_Density_init(self):
        """All measurements of the density without averaging."""
        dfn = self.data_pd_init.values
        # power hatching_distance scanning_velocity
        X = dfn[:, [1,2,3]]
        # density 
        Y = dfn[:, 7]
        return X, Y
    
    @property
    def X_Amorphisity_init(self):
        """All measurements of the amorphisity without averaging."""
        dfn = self.data_pd_init.values
        # power hatching_distance scanning_velocity
        X = dfn[:, [1,2,3]]
        # density 
        Y = dfn[:, 5]
        return X, Y

    def calc_power_density(self, p: float, h: float, v: float) -> float:
        """
        Calculates the energy density.
        Args:
            p: (float) laser power [W]
            h: (float) hatching distance [\mu m]
            v: (float) scanning velocity [m / s]
    
        Returns: 
            (float) p / (v * h) power density [MJ / m^2]
        """ 
        return p / (v * h)
    
    def calc_enthalpy(self, p: float, v: float) -> float:
        """ 
        Calculate the modified normalized enthalpy criterion.
        Args:
            p: (float) laser power [W]
            v: (float) scanning velocity [\mu m]
        
        Returns:
            (float) p / np.sqrt(v) normalized enthalpy criterion
        """
        return p / np.sqrt(v)
    
    def calc_mod_enthalpy_1(self, p: float, h: float, v: float) -> float:
        """
        Calculates the modified enthalpy.
        Args:
            p: (float) laser power [W]
            h: (float) hatching distance [\mu m]
            v: (float) scanning velocity [m / s]
        
        Returns:
            (float) p / ( h * np.sqrt(v)) modified enthalpy criterion
        """
        return p / ( h * np.sqrt(v) )

    def calc_mod_enthalpy_2(self, p: float, h: float, v: float) -> float:
        """
        Calculates the modified enthalpy.
        Args:
            p: (float) laser power [W]
            h: (float) hatching distance [\mu m]
            v: (float) scanning velocity [m / s]
        
        Returns:
            (float) p / ( h * np.sqrt(v)) modified enthalpy criterion
        """
        return p / ( np.sqrt(h * v) )


