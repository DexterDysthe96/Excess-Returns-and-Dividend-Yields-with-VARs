# Dexter Dysthe
# Dr. Bekaert
# B9325: Financial Econometrics, Time Series
# 2 May 2022

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from patsy import dmatrices
from statsmodels.tsa.api import VAR
from scipy import stats
from random import choices

np.random.seed(51796)
sns.set()

# ---------------------------------------------------- Question 1 ---------------------------------------------------- #

# ------------------------ Part 3 ------------------------ #

                                           # ------- Model w/o jumps ------- #

# The plot of returns using this model does not look anything like the data where by "the data" I am referring to the
# log returns data used to generate the plot on page 7 of the Lecture 1 notes. While the range of the log returns
# generated below account for the ~95% band of the data, they fail to account for the positive and negative spikes clearly
# apparent in the data.
log_rets_model1 = np.random.normal(0.0004, 0.01, 12500)

yrs_ticks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
yrs = np.linspace(1, 50, 12500)

plt.plot(yrs, log_rets_model1)
plt.xlabel('Years')
plt.xticks(yrs_ticks, yrs_ticks)
plt.ylabel('Daily Log Returns')
plt.title('Log Returns no jumps')
plt.show()


                                         # ------- Model w/ jumps ------- #

# While the plot of returns generated below using the jump model more accurately captures the data in that it accounts for
# negative spikes, it fails to account for the positive spikes clearly apparent in the data. One way to remedy this would be
# to add another jump term where we take the corresponding mu_J term to be positive; that is, mu_J of our single jump process
# is -0.03, and thus for the additional jump term we add in we would take the analogous parameter to be positive e.g. 0.03.
# We can do better. Notice in the data that the positive and negative spikes exhibit "clustering"; that is, the negative and
# positive spikes are not isolated but rather occur in tandem with one another (do positive spikes lead negative spikes or do
# negative spikes lead positive ones?). In the continuous time setting, one could accomplish building this into the model via
# Hawkes processes. In the current setting, let us suppose that negative spikes lead positive ones and let J_t,neg and J_t,pos
# denote the negative and positive jump processes respectively (negative/positive refers to the sign of mu_J) with p_neg and p_pos
# denoting the corresponding probability parameters of the underlying Bernoulli RVs B_t,neg and B_t,pos respectively. We induce
# correlation between B_t,neg and B_t,pos by letting p_pos be larger when B_t,neg = 1 and small when B_t,pos = 1. That is, let
# the probability parameter p_pos for B_t,pos depend on the outcome of B_t,neg. This is my attempt at discretizing the Hawkes
# process.
def jump_model(N, num_days, mu=0.0004, sigma=0.01, p=0.01, mu_J=-0.03, sigma_J=0.04):
    all_paths = list()
    for nn in range(N):
        delta_ts = np.random.normal(0, 1, num_days)
        ep_ts = np.random.normal(0, 1, num_days)
        b_ts = stats.bernoulli.rvs(size=num_days, p=p)

        jumps = np.asarray([b_ts[tt]*(mu_J + sigma_J*delta_ts[tt]) for tt in range(num_days)])
        log_rets = np.asarray([mu + sigma*ep_ts[tt] + jumps[tt] for tt in range(num_days)])
        all_paths.append(log_rets)

    return np.array(all_paths)


yrs_ticks2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
yrs2 = np.linspace(1, 10, 2500)

log_rets_model2 = jump_model(1, 2500).transpose()
plt.plot(yrs2, log_rets_model2)
plt.xlabel('Years')
plt.xticks(yrs_ticks2, yrs_ticks2)
plt.ylabel('Daily Log Returns')
plt.title('Log Returns with jumps')
plt.show()

# Sample moments
sample_mean_q1p3, sample_sd_q1p3, sample_skew_q1p3, sample_kurt_q1p3 = (np.mean(log_rets_model2),
                                                                        np.std(log_rets_model2, ddof=1),
                                                                        stats.skew(log_rets_model2),
                                                                        stats.kurtosis(log_rets_model2))
print('----------------- Q1P3 -----------------')
print('Sample mean: {}'.format(sample_mean_q1p3))
print('Sample standard deviation: {}'.format(sample_sd_q1p3))
print('Sample skew: {}'.format(sample_skew_q1p3))
print('Sample kurtosis: {}\n\n'.format(sample_kurt_q1p3))


# ------------------------ Part 4 ------------------------ #

def q3_p4(N, num_days, T):
    # Note to self: this function in more ways than one violates the single-responsibility principle.
    # I coded it this way hastily just to "big bazooka" this problem

    # Simulate N paths from our jump model each with num_days many days
    log_rets_sim_paths4 = jump_model(N, num_days)

    # Obtain means and standard errors of each of the N paths. Use 'A Note on Asymptotic Standard Errors"
    # from Canvas to obtain form for standard errors -- see write up for discussion.
    log_rets_means = np.asarray([np.mean(path) for path in log_rets_sim_paths4])
    log_rets_ses = np.asarray([np.std(path, ddof=1)/np.sqrt(num_days) for path in log_rets_sim_paths4])

    # Calculate t stats for each of the N paths
    log_rets_tstats = log_rets_means / log_rets_ses

    # ----------------------- Plotting ----------------------- #
    # Plot histogram of sample means
    plt.hist(log_rets_means, edgecolor='black', bins=math.ceil(np.sqrt(N)))
    plt.title('Histogram of Sample Means (T = {})'.format(T))
    plt.show()

    # Plot histogram of t statistics
    plt.hist(log_rets_tstats, edgecolor='black', bins=math.ceil(np.sqrt(N)))
    plt.title('Histogram of t Statistics (T = {})'.format(T))
    plt.show()

    # Calculate sample moments of the N many t stats
    sample_mean_tstats, sample_sd_tstats, sample_skew_tstats, sample_kurt_tstats = (np.mean(log_rets_tstats),
                                                                                    np.std(log_rets_tstats, ddof=1),
                                                                                    stats.skew(log_rets_tstats),
                                                                                    stats.kurtosis(log_rets_tstats))
    print('------------------------------ T = {} ------------------------------'.format(T))
    print('------------ t statistics ------------')
    print('Sample mean: {}'.format(sample_mean_tstats))
    print('Sample standard deviation: {}'.format(sample_sd_tstats))
    print('Sample skew: {}'.format(sample_skew_tstats))
    print('Sample kurtosis: {}\n'.format(sample_kurt_tstats))


    # Calculate sample moments of the N many sample means
    sample_mean_means, sample_sd_means, sample_skew_means, sample_kurt_means = (np.mean(log_rets_means),
                                                                                np.std(log_rets_means, ddof=1),
                                                                                stats.skew(log_rets_means),
                                                                                stats.kurtosis(log_rets_means))
    print('------------ sample means ------------')
    print('Sample mean: {}'.format(sample_mean_means))
    print('Sample standard deviation: {}'.format(sample_sd_means))
    print('Sample skew: {}'.format(sample_skew_means))
    print('Sample kurtosis: {}\n'.format(sample_kurt_means))

    # The difference between the below two values is approximately monotonically decreasing in T. For T = 4, the
    # difference between the two is roughly 0.00001 whereas for T = 48 the difference is roughly 0.0000005.
    avg_se = np.mean(log_rets_ses)
    sd_of_sample_means = np.std(log_rets_means, ddof=1)
    print('Average of standard errors: {}'.format(avg_se))
    print('Standard deviation of sample means: {}\n'.format(sd_of_sample_means))
    print('-------------------------------------------------------------------\n')


# *** Since it is not specified, I will take N = 1000 *** #
# T = 4
q3_p4(1000, 1000, 4)
# T = 8
q3_p4(1000, 2000, 8)
# T = 12
q3_p4(1000, 3000, 12)
# T = 24
q3_p4(1000, 6000, 24)
# T = 48
q3_p4(1000, 12000, 48)


# ---------------------------------------------------- Question 2 ---------------------------------------------------- #
hwk4_df = pd.read_csv('homework4data.csv')

# ------------------------ Part 1 ------------------------ #
hwk4_df['log_excess_ret_times12'] = 12 * hwk4_df['log_excess_ret']

hwk4_df_q2p1 = pd.DataFrame({'log_excess_ret_times12': np.asarray(hwk4_df['log_excess_ret_times12'].loc[1:].reset_index(drop=True)),
                            'div_yld': np.asarray(hwk4_df['div_yld'].iloc[:-1].reset_index(drop=True))})
y_q2p1, X_q2p1 = dmatrices('log_excess_ret_times12 ~ div_yld', data=hwk4_df_q2p1, return_type='dataframe')
q2p1_model = sm.OLS(y_q2p1, X_q2p1)

# Using White standard errors
q2p1_res = q2p1_model.fit(cov_type='HC1')
print('----------------------------------------- Q2 Part 1 -----------------------------------------')
print('\n')

# As can be seen from the output below, the p-value on the coefficient for dividend yield for the null that Beta = 0
# is equal to 0.263, and thus we fail to reject the null hypothesis at all conventional levels
print(q2p1_res.summary(), '\n')


# ------------------------ Part 2 ------------------------ #
hwk4_df_q2p2 = pd.DataFrame({'sum_of_rets': hwk4_df['log_excess_ret'].rolling(12).sum().iloc[12:].reset_index(drop=True),
                             'div_yld': hwk4_df['div_yld'].iloc[:-12].reset_index(drop=True)})
print(hwk4_df_q2p2.isnull)
y_q2p2, X_q2p2 = dmatrices('sum_of_rets ~ div_yld', data=hwk4_df_q2p2, return_type='dataframe')
q2p2_model = sm.OLS(y_q2p2, X_q2p2)

# Using White standard errors
q2p2_res1 = q2p2_model.fit(cov_type='HC1')

print('----------------------------------------- Q2 Part 2 -----------------------------------------')
print('\n')

# Conduct a Ljung-Box test to test whether residuals are autocorrelated. Use the rule of taking m equal to the integer
# ceiling of the natural log of the length of the vector of residuals. We obtain a p_value of 0.0 for this test, and this
# is robust to other values of the m-parameter for the LB test. Thus, we reject the null that the first m coefficients of
# the ACF are equal to zero at all conventional levels. As a result, the residuals follow some serially correlated process.
ljung_box = sm.stats.acorr_ljungbox(q2p2_res1.resid, lags=[math.ceil(np.log(len(list(q2p2_res1.resid))))], return_df=True)
print('Ljung-Box test statistic: \n', ljung_box)

# Given the results of the above LB test, we rerun the regression with heteroskedasticity and autocorrelation robust standard
# errors. We set maxlags equal to 3 following the convention Dr. Bekaert discussed in class, however, we acknowledge the point
# made that recently this has been called into question and that Dr. Bekaert now also considers larger values for the lag parameter
# for Newey-West SEs in his own work.
q2p2_res = q2p2_model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
print(q2p2_res.summary(), '\n')


# ------------------------ Part 3 ------------------------ #
hwk4_df_q2p3 = hwk4_df[['log_excess_ret', 'div_yld']]
q2p3_model = VAR(hwk4_df_q2p3)
bic = q2p3_model.select_order(3)
print('----------------------------------------- Q2 Part 3 -----------------------------------------')
bic_df = pd.DataFrame(bic.summary()).drop([1], axis=0).reset_index().iloc[:, [0, 3]]
bic_df.columns = bic_df.iloc[0]
bic_df = bic_df.iloc[1:, :]
print(bic_df.iloc[:, 1:])
print('\n')

opt_BIC_var = q2p3_model.fit(1)
print(opt_BIC_var.summary())


# ------------------------ Part 4 ------------------------ #
# Q1. Using the default standard errors from the statsmodels.tsa.api library
var_q1 = VAR(hwk4_df_q2p1).fit(1)
print(var_q1.summary())

# Q2. Using the default standard errors from the statsmodels.tsa.api library
var_q2 = VAR(hwk4_df_q2p2).fit(1)
print(var_q2.summary())


# ------------------------ Part 5 ------------------------ #
hwk4_df_q2p5 = hwk4_df[['log_excess_ret', 'div_yld']]

# Regress return onto a constant and store coefficient
alpha_ret = hwk4_df_q2p5['log_excess_ret'].mean()
# Remove first residual so that the size matches with the dy regression residuals
ret_resids = np.asarray(hwk4_df_q2p5['log_excess_ret'] - alpha_ret)[1:]

# Create dependent variable and lag for independent
div_yld_t = np.asarray(hwk4_df_q2p5['div_yld'].iloc[1:])
div_yld_tlag = np.asarray(hwk4_df_q2p5['div_yld'].iloc[:-1])

# Regress dividend yield on a constant and one lag
beta_dy = np.cov(div_yld_t, div_yld_tlag)[0][1] / np.var(div_yld_tlag)
alpha_dy = div_yld_t.mean() - beta_dy * div_yld_tlag.mean()
dy_resids = div_yld_t - (alpha_dy + beta_dy * div_yld_tlag)

print('----------------------------------------- Q2 Part 5 -----------------------------------------')

# Correlation matrix between regression residuals. As can be seen by the off diagonal entries,
# the residuals from the return regression and the residuals from the dividend yield regression
# are highly negatively correlated w/ one another.
print('Correlation matrix of residuals: \n', np.corrcoef(ret_resids, dy_resids))

# Join the two lists of residuals
vec_of_resids = np.asarray(list(zip(ret_resids, dy_resids)))
tstats_q1 = list()
tstats_q2 = list()
rsquared_q2 = list()
for bb in range(1000):
    # Draw residuals vectors with replacement
    bootstrapped_resids = choices(vec_of_resids, k=len(vec_of_resids))

    # Reconstruct returns
    bootstrapped_returns = pd.Series([alpha_ret + bootstrapped_resids[tt][0] for tt in range(len(bootstrapped_resids))])

    # Run regression for Q1 using reconstructed returns
    q1_boot = pd.DataFrame({'log_excess_ret_times12': 12 * bootstrapped_returns,
                            'div_yld': hwk4_df_q2p5['div_yld'].iloc[:-1]})
    y_q1boot, X_q1boot = dmatrices('log_excess_ret_times12 ~ div_yld', data=q1_boot, return_type='dataframe')
    q1boot_model = sm.OLS(y_q1boot, X_q1boot)
    q1boot_res = q1boot_model.fit(cov_type='HC1')
    tstats_q1.append((q1boot_res.params / q1boot_res.bse).values[1])

    # Run regression for Q2 using reconstructed returns
    q2_boot = pd.DataFrame({'sum_of_rets': bootstrapped_returns.rolling(12).sum().iloc[11:],
                            'div_yld': hwk4_df['div_yld'].iloc[:-11]})
    y_q2boot, X_q2boot = dmatrices('sum_of_rets ~ div_yld', data=q2_boot, return_type='dataframe')
    q2boot_model = sm.OLS(y_q2boot, X_q2boot)
    q2boot_res = q2boot_model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    tstats_q2.append((q2boot_res.params / q2boot_res.bse).values[1])
    rsquared_q2.append(q2boot_res.rsquared)


# Plot empirical distribution of t statistics for Q1 and empirical distributions of t statistics and R^2 for Q2
plt.hist(tstats_q1, edgecolor='black', bins=math.ceil(np.sqrt(1000)))
plt.title('Histogram of t Statistics (Q1)')

# The critical values for a 5% two-sided t test are roughly 1.97 and -1.97. This test rejects the null with significantly
# smaller probability than the t test for the Q2 model, however, this does not necessarily indicate that the probability of a
# Type 1 error (the size of the test) is also smaller. Nonetheless, it is very likely the case that the size of this test
# is smaller than that of the test used below for the Q2 model.
plt.axvline(x=1.97, color='red')
plt.axvline(x=-1.97, color='red')
plt.show()

plt.hist(tstats_q2, edgecolor='black', bins=math.ceil(np.sqrt(1000)))
plt.title('Histogram of t Statistics (Q2)')
# The critical values for a 5% two-sided t test are roughly 1.97 and -1.97.
plt.axvline(x=1.97, color='red')
plt.axvline(x=-1.97, color='red')
plt.show()

plt.hist(rsquared_q2, edgecolor='black', bins=math.ceil(np.sqrt(1000)))
plt.title('Histogram of R Squared Value (Q2)')
plt.show()


# ------------------------ Part 6 ------------------------ #

rolling_betas = list()
ci_rolling_betas = list()
# The excel has 720 rows, and thus we roll through until the 601st row which will
# constitute the last 10 year window
for roll_start in range(601):
    # Obtain 10 years worth of data
    roll_period = hwk4_df.iloc[roll_start:roll_start+120]

    # Create necessary columns to replicate Q1 regression on the current window of data
    hwk4_df_q2p6 = pd.DataFrame({'log_excess_ret_times12': 12 * np.asarray(roll_period['log_excess_ret'].iloc[1:]),
                                 'div_yld': np.asarray(roll_period['div_yld'].iloc[:-1])})
    y_q2p6, X_q2p6 = dmatrices('log_excess_ret_times12 ~ div_yld', data=hwk4_df_q2p6, return_type='dataframe')
    q2p6_model = sm.OLS(y_q2p6, X_q2p6)

    # Using White standard errors
    q2p6_res = q2p6_model.fit(cov_type='HC1')

    # Append estimated dividend yield beta and confidence interval radius
    rolling_betas.append(q2p6_res.params[1])
    rolling_beta_se = q2p6_res.bse.values[1]
    ci_rolling_betas.append(stats.t.ppf(q=.975, df=119) * rolling_beta_se)


plt.plot(rolling_betas)
plt.plot(np.asarray(rolling_betas) - np.asarray(ci_rolling_betas), color='blue', linestyle='dashed')
plt.plot(np.asarray(rolling_betas) + np.asarray(ci_rolling_betas), color='blue', linestyle='dashed')
plt.title('Estimated Rolling Window Dividend Yield Coefficients')
plt.ylabel('Betas')
plt.xlabel('Window')
plt.show()


# ------------------------ Part 7 ------------------------ #
hwk4_df_q2p7 = pd.DataFrame({'log_nfirms': np.log(hwk4_df['nfirms'].iloc[:-1].reset_index(drop=True)),
                             'minus_log_mktcap': -np.log(1000 * hwk4_df['me'].iloc[:-1].reset_index(drop=True)),
                             'log_excess_ret_times12': hwk4_df['log_excess_ret_times12'].loc[1:].reset_index(drop=True)})
y_q2p7, X_q2p7 = dmatrices('log_excess_ret_times12 ~ log_nfirms + minus_log_mktcap', data=hwk4_df_q2p7, return_type='dataframe')
q2p7_model = sm.OLS(y_q2p7, X_q2p7)
# Using White standard errors
q2p7_res = q2p7_model.fit(cov_type='HC1')
print('----------------------------------------- Q2 Part 7 -----------------------------------------')
print('\n')

# This regression is not well-specified as it suffers from an omitted variables problem since annual dividend
# yield is excluded which has a correlation of -0.7066 with the natural logarithm of total market cap. Because
# of this, the OLS estimates are biased and inconsistent.
print(q2p7_res.summary(), '\n')
