# Sample GitLab Project

This sample project shows how a project in GitLab looks for demonstration purposes. It contains issues, merge requests and Markdown files in many branches,
named and filled with lorem ipsum.

You can look around to get an idea how to structure your project and, when done, you can safely delete this project.

[Learn more about creating GitLab projects.](https://docs.gitlab.com/ee/gitlab-basics/create-project.html)

# Getting Started

You can install CalibLab via downloading the zip file or either clone this repository.


The user must provide the following files in the "data" folder:
- **Metered data** ("HeatEnergyDemand_30034001_Y.xlsx")
- **Degree of uncertainty of model parameters** ("prior_probability_definition.xlsx")
- **Fixed model parameters for DIBS** ("fixed_variables.xlsx")
- **Prior model parameter values** ("variables_df.xlsx")

<br>The framework is run from the main file "src/main.py"

Required inputs:
- **Building ID**
- **Time resolution**: Yearly ('Y'), Monthly ('M'), Weekly ('W') etc.
- **Climate data type**: Actual Meteorological Year ('AMY') or Test Reference Year, Potsdam ('TRY')
- **Climate file**: File name
- **Number of calibration parameters**

Optional inputs:
- Request for automatized convergence search for Sensitivity Analysis
- Request for automatized convergence search for Gaussian Processes
- Percentage of metered data used for the calibration



# Main  Python packages:
Basic packages:
- [Pandas](https://pypi.org/project/pandas/)
- [Numpy](https://pypi.org/project/numpy/)

DIBS:
- [Namedlist](https://pypi.org/project/namedlist/)
- [Geopy](https://pypi.org/project/geopy/)

Sensitivity Analysis:
- [SALib](https://salib.readthedocs.io/en/latest/)

Gaussian Processes:
- [Scikit-learn](https://scikit-learn.org/stable/)

Bayesian Inference:
- [PyMC3](https://pypi.org/project/pymc3/)
- [Arviz](https://python.arviz.org/en/stable/)
- [Theano](https://pypi.org/project/Theano/)



# Publications

Fülep, Katalin; Kalathiparambil Kennedy, Mary Lidiya; Brandt, Stefan; Streblow, Rita (2024): An Open-Source Framework: Bayesian Building Energy Model Calibration with Uncertainty Quantification. In the proceedings of the uSIM2024 in Edinburgh, 25. November 2024.

Fülep, Katalin; Chen, Siling; Brandt, Stefan; Streblow, Rita (2024): Advancing Building Energy Modeling: An Open-Source Bayesian Calibration Framework for Non-Residential Buildings. In the proceedings of the SimBuild in Denver, Colorado, 21-23 May 2024. In SimBuild 2024 URL: https://publications.ibpsa.org/conference/paper/?id=simbuild2024_2292

---
The CalibLab has been developed in context of the [BaU-SaN project](https://www.zukunftbau.de/projekte/forschungsfoerderung/1008187-2137zukunftbau
) funded by Forschungsinitiative Zukunft Bau des Bundesinstitutes für Bau-, Stadt- und Raumforschung (reference number SWD-10.08.18.7-21.37). ZUKUNFT BAU [www.zukunftbau.de](https://www.zukunftbau.de/)


!(ZukunftBau.jpg)


