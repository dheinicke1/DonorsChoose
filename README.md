# Donors Choose Challenge

[Kaggle Donors Choose Competition](https://www.kaggle.com/c/donorschoose-application-screening)

[Data Here](https://www.kaggle.com/c/donorschoose-application-screening/data)

The code for the various models are in the ["Previous Versions"]() folder, the code used for my final submission is
["DonorsChoose.py"]()

## Objective

[Donors Choose](https://www.donorschoose.org/about) is a nonprofit organization that pairs teachers throughout the United States who have classroom projects with donors wiling to fund those projects.

The teachers submit an application to Donors Choose that includes a description, application essays, a list of supplies requested, teacher attributes and school attributes. The purpose of the competition is to use these text, categorical and numeric features to predict whether a project will be funded. The prediction is used to assist volunteers for Donors Choose with their project vetting process and allowing volunteers to assist teachers with their applications to increase their chances of a successfully funded project.

Competition submissions are a list of probabilities of the application being approved, rather than a binary prediction (approved or not approved). Submissions are scored on the area under the AUC curve.

The winner achieved a score of 0.828, my model achieved a score of 0.779.

## Methodology

### EDA

Before jumping into what makes for a good application, it is worth getting an idea of what the applications look like, what the teachers are requesting, where the applications are coming from and so on. This will help guide the feature extraction later on.

There are a number of features to explore  including the grade being taught, the teacher's application history and details about the project, but here are a few examples to start with.

First, are the project categories well balanced, or are others more popular than others?

<p align="center">
  <img width="514" height="439" src="https://github.com/dheinicke1/DonorsChoose/blob/master/files/Category_Instances.png">
</p>

How much are the teachers requesting, and how are the requests distributed?

<p align="center">
  <img width="576" height="396" src="https://github.com/dheinicke1/DonorsChoose/blob/master/files/ProjectCosts.png">
</p>

The vast majority of projects are under $200, although there are some projects in the thousands of dollars (not shown). The second plot shows the success rate by cost-grouping, with the average project costs requested split into 100 equally-sized groups (using adaptive binning). Interestingly, both teachers who ask for the least amount and teachers who ask for the most achieved higher success rates.

How are the applications distributed geographically? Do some states have a disproportionate number of applications relative to their population? Is there a relationship between state income and the grant application rate?

For this, I had to bring in external census data.

<p align="center">
  <img width="513" height="365" src="https://github.com/dheinicke1/DonorsChoose/blob/master/files/population_vs_applications.png">
</p>

There are some outliers - South Carolina has a disproportionately high number of applications, while the number of applications in Texas is disproportionately low. Its not clear why, but maybe modeling will reveal differences by state.

What about state income vs number of applications? Does state income predict applications? Apparently no!

<p align="center">
  <img width="506" height="366" src="https://github.com/dheinicke1/DonorsChoose/blob/master/files/income_vs_application.png">
</p>

Here's my takeaway for bringing in external state data:

*There is variation by state in participation in the program, but sates are big. There is a lot of variation in schools within states, so its unlikely that just taking a state summary will tell us much. There could be a relationship between income in a school district and participation and success in the program, but we would need to join a more granular dataset to bring this into a model.*
