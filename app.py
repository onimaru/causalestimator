import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from time import time

import dowhy
from dowhy import CausalModel

st.set_page_config(layout="wide")

# Avoid printing dataconversion warnings from sklearn
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Config dict to set the logging level
import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'WARN',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)

st.markdown("""
# Causal Effect Estimator

---
""")


class InferCausalModel:
    def __init__(self,data_df,params):
        self.treatment           = params["treatment"]
        self.outcome             = params["outcome"]
        self.common_causes       = params["common_causes"]
        self.effect_modifiers    = params["effect_modifiers"]
        self.instruments         = params["instruments"]
        self.dag                 = params["dag"]
        self.df                  = self.build_dag_df(data_df)
        self.model               = self.build_model()
        self.identified_estimand = self.model.identify_effect(proceed_when_unidentifiable=True)

    def build_dag_df(self,data_df):
        cols = [self.treatment]+[self.outcome]+self.common_causes+self.effect_modifiers+self.instruments
        df = data_df[cols].copy()
        df[self.treatment] = df[self.treatment].astype(bool)
        return df

    def build_model(self):
        model = CausalModel(data             = self.df,
                            treatment        = self.treatment,
                            outcome          = self.outcome,
                            common_causes    = self.common_causes,
                            effect_modifiers = self.effect_modifiers,
                            instruments      = self.instruments,
                            graph            = self.dag.replace("\n", " "))
        return model

    def compute_estimates(self):
        self.estimate     = self.model \
                                .estimate_effect(self.identified_estimand,
                                                 method_name="backdoor.propensity_score_stratification")
        self.estimate_atc = self.model \
                                .estimate_effect(self.identified_estimand,
                                                 method_name="backdoor.propensity_score_stratification",
                                                 target_units = "atc")
        self.estimate_att = self.model \
                                .estimate_effect(self.identified_estimand,
                                                 method_name="backdoor.propensity_score_stratification",
                                                 target_units = "att")

    def compute_refuters(self):
        self.res_random  = self.model.refute_estimate(self.identified_estimand,
                                            self.estimate,
                                            method_name="random_common_cause")
        self.res_placebo = self.model.refute_estimate(self.identified_estimand,
                                            self.estimate,
                                            method_name="placebo_treatment_refuter",
                                            placebo_type="permute")
        self.res_subset  = self.model.refute_estimate(self.identified_estimand,
                                            self.estimate,
                                            method_name="data_subset_refuter",
                                            subset_fraction=0.9)
    
    def compute_estimates_and_refuters(self):
        self.compute_estimates()
        self.compute_refuters()

    def results_summary(self):
        self.summary=f"""
**Efeitos**:

| ATE | ATC | ATT |
| --- | --- | --- |
| {self.estimate.value:.3g} | {self.estimate_atc.value:.3g} | {self.estimate_att.value:.3g}|

**Explicações**:  
**ATE** - Aumentando o tratamento da variável `{self.treatment}` de 0 para 1 causa um aumento de **{self.estimate.value:.3g}** no valor esperado do resultado `{self.outcome}`, sobre a população representada no dataset.

**ATC** - Aumentando o tratamento da variável `{self.treatment}` de 0 para 1 causa um aumento de **{self.estimate_atc.value:.3g}** no valor esperado do resultado `{self.outcome}`, sobre o `grupo de controle`.

**ATT** - Aumentando o tratamento da variável `{self.treatment}` de 0 para 1 causa um aumento de **{self.estimate_att.value:.3g}** no valor esperado do resultado `{self.outcome}`, sobre `grupo de tratamento`.

**Refutadores**:  
Random e Subset - Se as suposições estiverem corretas o novo valor não deve mudar muito.  
Placebo - Se as suposições estiverem corretas o novo valor deve ser próximo de zero.  

| Efeito estimado | Efeito random | Efeito Subset |Efeito Placebo |
| --- | --- | --- | --- |
| {self.estimate.value:.3g} | {self.res_random.new_effect:.3g} |{self.res_subset.new_effect:.3g} | {self.res_placebo.new_effect:.3g} |
"""
    
    def show_summary(self):
        self.results_summary()
        return self.summary

def show_dataframe(data_file):
    data_df = pd.read_csv(data_file)
    st.dataframe(data_df)
    return True, data_df

my_expander = st.sidebar.beta_expander(label='How to')
with my_expander:
    st.markdown("""
- Construa a dag em http://dagitty.net/dags.html (com os mesmos nome usados no seu dataset)
- Copie o conteúdo de `Model code` e cole no campo `Grafo`, deixando com o formato abaixo do exemplo
- No campo `Params` informe:
  - a variável de tratamento em `treatment`
  - a variável resultado em `outcome`
  - as features que são causas comuns em `common_causes`
  - variáveis modificadoras de efeito em `effect_modifiers`
  - variáveis instrumentais em `instruments`  
Obs: cada feature só pode aparecer uma vez em cada um dos campos de `Params`
    """)

dag = st.sidebar.text_area("Grafo",height=700,value="digraph {\nX -> Y;\nZ1 -> Y;\nZ1 -> X;\nZ2 -> Y;\nZ2 -> X;\n}")
default_params = """{'treatment': 'X',
'outcome': 'Y',
'common_causes': ['Z1','Z2'],
'effect_modifiers': [],
'instruments': []}"""
params = st.text_area('Params',height=180,value=default_params)
params = eval(params)
params["dag"] = dag
data_file = st.file_uploader('Dataset')
showed_df = False
if st.button("Show dataframe"):
    showed_df, data_df = show_dataframe(data_file)

if st.button('Run analysis'):
    begin = time()
    if showed_df == False:
        data_df = pd.read_csv(data_file)
    causal_model = InferCausalModel(data_df,params)
    causal_model.compute_estimates_and_refuters()
    st.markdown(f"**Tempo de execução**: {(time()-begin)/60:.2} min.")
    st.markdown(causal_model.show_summary())