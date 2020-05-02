from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process


class ActiveGridLayout(GridLayout):
    gold_model = ''
    active_result = []
    @staticmethod
    def get_path():
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()

        return filedialog.askopenfilename()

    @staticmethod
    def visualize_dataframe(filename):
        import PySimpleGUI as sg
        # Header=None means you directly pass the columns names to the dataframe
        df = pd.read_csv(filename, sep=',', engine='python', header=None)
        data = df[1:].values.tolist()
        header_list = df.iloc[0].tolist()
        # given a pandas dataframe
        layout = [[sg.Table(values=data, max_col_width=5,
                            auto_size_columns=True,
                            vertical_scroll_only=False,
                            justification='right', alternating_row_color='blue',
                            key='_table_', headings=header_list)]]

        window = sg.Window('Table', layout)
        event, values = window.read()
        if event is None or event == 'Back':
            window.close()

    def train_gold_model(self, filename):
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor
        # dataset
        dataset = pd.read_csv(filename)
        X = dataset.iloc[:, 0:11].values
        y = dataset.iloc[:, 11].values
        # Define our GOLD STANDARD MODEL
        gold_regressor = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8
        )
        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Validation Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        # Training ...
        gold_regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', early_stopping_rounds=15,verbose=False)
        # Predicting the Test set results
        y_pred = gold_regressor.predict(X_test)
        # save model
        self.gold_model = gold_regressor
        # Prediction Metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_pred, y_test)
        r2 = r2_score(y_pred, y_test)
        popup = Popup(title='Gold Standard Model', content=Label(text='Gold Model has been trained \n mean absolute '+
                                                                      'error = {} \n R2 = {}'.format(mae, r2)),
                      auto_dismiss=True,
                      size_hint=(None, None), size=(400, 400))
        popup.open()

    def acive_learning_main(self, steps_number=10):
        import numpy as np
        from xgboost import XGBRegressor
        gold_regressor = self.gold_model

        # Part 1: choose grid for our metabolite conc

        # Allowed concentrations
        allowed_conc = {
            'nad': (0.033, 0.33),
            'folinic_acid': (0.0068, 0.068),
            'coa_conc': (0.026, 0.26),
            'nucleo_mix': (0.15, 1.5),
            'spermidine': (0.1, 1.0),
            'pga': (3.0, 30.0),
            'aa': (0.15, 1.5),
            'trna': (0.02, 0.2),
            'mg_gluta': (0.4, 4.0),
            'camp': (0.075, 0.75),
            'K_gluta': (8.0, 80.0)}

        # Part 2: make a random input for our model
        def random_input(allowed_conc, n=100, rounded=3, verbose=0):
            X_train = []
            for data_point in range(n):
                input_data = []
                if (data_point % 10000 == 0) and verbose:
                    print(data_point)
                for key, value in allowed_conc.items():
                    input_data.append(np.round(np.random.uniform(*value), rounded))
                X_train.append(input_data)

            X_train = np.array(X_train)
            return X_train

        # define our model that will be trained by active learning
        # same hyperparameter as Gold Standard model
        regressor = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8)

        def active_learning(regressor, gold_regressor, allowed_conc, test_size=100, steps=10):
            ## first step
            # make first dataset
            X_train_1 = random_input(allowed_conc, test_size)

            # first fit
            regressor.fit(X_train_1, gold_regressor.predict(X_train_1))

            # save results
            result = pd.DataFrame(X_train_1)
            result['gold_yield'] = gold_regressor.predict(X_train_1)
            result['step'] = 'step_1'

            ## next steps loop
            for step in range(steps - 1):
                print('step: ', step)
                # make i th dataset
                X_train_1_1 = random_input(allowed_conc, 100000)
                df_1 = pd.DataFrame(X_train_1_1)
                df_1['pred_yield'] = regressor.predict(X_train_1_1)
                df_1 = df_1.sort_values(['pred_yield'], ascending=False)
                X_train_2 = df_1.iloc[0:test_size, 0:11].values

                # save and add results
                temp_result = pd.DataFrame(X_train_2)
                temp_result['gold_yield'] = gold_regressor.predict(X_train_2)
                temp_result['step'] = 'step_{}'.format(step + 2)
                result = pd.concat([result, temp_result], ignore_index=True)

                # update and refit regressor
                regressor.fit(result.iloc[:, 0:11].values, result.iloc[:, 11].values)

            popup = Popup(title='Active Learning',
                          content=Label(text='Active Learning Finished'),
                          auto_dismiss=True,
                          size_hint=(None, None), size=(400, 400))
            popup.open()
            return result, regressor

        self.active_result, _ = active_learning(regressor, gold_regressor, allowed_conc, steps=steps_number)

    @staticmethod
    def boxplot(data, group_name, quantity, title='', point_size=10):
        import seaborn as sns
        plt.figure(figsize=(6, 4))
        plt.style.use('seaborn-whitegrid')
        plt.style.use('seaborn-poster')
        # Usual boxplot
        sns.boxplot(x=group_name, y=quantity, data=data)
        # Add jitter with the swarmplot function.
        sns.swarmplot(x=group_name, y=quantity, data=data, color='k', size=point_size)
        plt.title(title)
        plt.show()

    def show_plot(self):
        t = Process(target=self.boxplot, args=(self.active_result, 'step', 'gold_yield', 'Kivy_BoxPlot', 5))
        t.start()


class ActiveApp(App):
    title = 'Active Learning'
    icon = 'icon.png'

    def build(self):
        return ActiveGridLayout()


if __name__ == '__main__':
    ActiveApp().run()
