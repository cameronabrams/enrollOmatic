import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title='Enroll-O-Matic', layout='centered', initial_sidebar_state='expanded')

# College Class Definition
class College:
    def __init__(self, config):
        self.C = np.array(config['initial_cohorts']).reshape(len(config['initial_cohorts']), 1)
        self.r = np.array(config['initial_retention_rates'])
        self.dr = np.array(config['retention_rate_uncertainties'])
        self.dC = config['new_enrollment_uncertainty']
        self.DeltaE = config['slope']
        self.duration = config['duration']
        self.nyears = len(self.C)
        self.cm = cm[config['colormapname']]
    
    def E(self):
        return self.C[:, -1].sum()
    
    def __next__(self):
        for i in range(self.nyears - 1):
            self.r[i] += norm.rvs(loc=0.0, scale=self.dr[i])
            self.r[i] = min(self.r[i], 1.0)
        eC = norm.rvs(loc=0.0, scale=self.dC)
        En = self.E()
        tC = np.zeros(self.nyears)
        tC[1:] = self.C[:, -1][:-1] * self.r
        tC[0] = self.DeltaE + En - tC.sum() + eC
        self.C = np.concatenate((self.C, np.array([tC]).reshape(-1, 1).round(0)), axis=1)
    
    def run(self):
        for _ in range(self.duration):
            next(self)
        x = np.arange(self.duration + 1).reshape(-1, 1)
        y = self.C.sum(axis=0)
        model = LinearRegression()
        model.fit(x, y)
        self.slope = int(model.coef_[0])
        self.r2 = model.score(x, y)
        self.y_pred_1 = model.predict(x)
        return self
    
    def stackplot(self):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.stackplot(range(self.C.shape[1]), self.C, labels=[str(i+1) for i in range(self.nyears)],
                     colors=[self.cm(x / self.C.shape[0]) for x in range(self.C.shape[0])], alpha=0.5)
        ax.legend()
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Enrollment')
        ax.set_title(f'R²={self.r2:.3f}, Slope={self.slope}')
        st.pyplot(fig)

# Streamlit UI
def main():
    st.title('📊 Enroll-O-Matic')
    
    st.markdown("""
    ## Welcome to the Enrollment Simulation App
    
    This application allows you to model enrollment changes over time for a hypothetical college. 
    You can adjust various parameters, including initial cohort sizes, retention rates, and 
    uncertainties to explore different scenarios.
    """)
    
    st.sidebar.header('📌 Simulation Parameters')
    
    num_cohorts = st.sidebar.number_input('Number of Cohorts (4 or more)', min_value=4, value=4, step=1)
    
    initial_cohorts = [st.sidebar.number_input(f'Year {i+1} Cohort', min_value=0, value=1000) for i in range(num_cohorts)]
    retention_rates = [st.sidebar.slider(f'Retention Year {i+1}', 0.0, 1.0, 0.88, 0.01) for i in range(num_cohorts - 1)]
    retention_uncertainties = [st.sidebar.slider(f'Uncertainty Year {i+1}', 0.0, 0.1, 0.02, 0.01) for i in range(num_cohorts - 1)]
    new_enrollment_uncertainty = st.sidebar.slider('New Enrollment Uncertainty', 0, 100, 40)
    slope = st.sidebar.number_input('Enrollment Slope', -500, 500, -170)
    duration = st.sidebar.number_input('Simulation Duration (years)', 5, 50, 20)
    colormapname = st.sidebar.selectbox('Colormap', ['viridis', 'plasma', 'winter'])
    
    config = {
        'initial_cohorts': initial_cohorts,
        'initial_retention_rates': retention_rates,
        'retention_rate_uncertainties': retention_uncertainties,
        'new_enrollment_uncertainty': new_enrollment_uncertainty,
        'slope': slope,
        'duration': duration,
        'colormapname': colormapname
    }
    
    if st.sidebar.button('Run Simulation'):
        C = College(config).run()
        C.stackplot()
    
    st.markdown("""
    ### Understanding the Simulation
    - The **slope** parameter represents the overall trend in enrollment change.
    - **Retention rates** impact how many students continue each year.
    - **Uncertainty factors** introduce randomness into the predictions.
    """)

if __name__ == '__main__':
    main()
