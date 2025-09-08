# --- 1. Calculate the Data-Driven Stress Factor ---
# (This part of your code is correct and remains the same)
regime_vols = final_hmm_model.means_[:, 1]
v_calm = regime_vols.min()
v_crisis = regime_vols.max()
regime_stress_weights = (regime_vols - v_calm) / (v_crisis - v_calm)
stress_factor = prob_df.dot(regime_stress_weights)
stress_factor.name = "StressFactor"

# --- 2. Build the Hybrid Model with Non-Linear Blending ---
hybrid_df = pd.DataFrame(index=stress_factor.index)
hybrid_df['StressFactor'] = stress_factor
hybrid_df['VaR_GJR'] = gjr_garch_var_99
hybrid_df['VaR_EVT'] = evt_var_99_series
hybrid_df.dropna(inplace=True)

# --- THE FIX: Apply a non-linear transformation to the stress factor ---
# Squaring the factor makes the model much stricter.
strict_stress_factor = hybrid_df['StressFactor']**2

# Apply the new blending formula
hybrid_df['VaR_Hybrid'] = (
    (1 - strict_stress_factor) * hybrid_df['VaR_EVT'] +
    strict_stress_factor * hybrid_df['VaR_GJR']
)

print("\n✅ Final non-linear hybrid VaR calculation complete.")

# --- 3. Visualization ---
# (The plotting code is the same, but we add the new stress factor for comparison)
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 12), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)
fig.suptitle('Final Hybrid VaR with Non-Linear Blending', fontsize=20)

ax1.plot(portfolio_returns, color='gray', alpha=0.3, label='Portfolio Returns')
ax1.plot(hybrid_df['VaR_GJR'], color='purple', linestyle=':', alpha=0.5, label='Component: GJR-GARCH VaR')
ax1.plot(hybrid_df['VaR_EVT'], color='red', linestyle=':', alpha=0.5, label='Component: EVT VaR')
ax1.plot(hybrid_df['VaR_Hybrid'], color='blue', linewidth=2.5, label='Final Hybrid VaR')
ax1.set_title('Final Blended VaR vs. Component Models', fontsize=16)
ax1.set_ylabel('Daily Return', fontsize=12)
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)

# Plot both the original and the new, stricter stress factor
ax2.plot(hybrid_df['StressFactor'], color='gray', linestyle=':', alpha=0.7, label='Original Linear Factor')
ax2.plot(strict_stress_factor, color='teal', linewidth=2, label='Strict Non-Linear Factor')
ax2.set_title('Market Stress Factor (Weight on GJR-GARCH Model)', fontsize=16)
ax2.set_ylabel('Stress Factor', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()



# --- Build and Visualize the Final Hybrid Model ---
hybrid_df = pd.DataFrame(index=stress_factor.index)
hybrid_df['StressFactor'] = stress_factor
hybrid_df['VaR_GJR'] = gjr_garch_var_99
hybrid_df['VaR_EVT'] = evt_var_99_series
hybrid_df.dropna(inplace=True)

hybrid_df['VaR_Hybrid'] = (
    (1 - hybrid_df['StressFactor']) * hybrid_df['VaR_EVT'] +
    hybrid_df['StressFactor'] * hybrid_df['VaR_GJR']
)
print("\n✅ Step 3 complete: Hybrid VaR calculation finished.")

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 12), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)
fig.suptitle('Final Hybrid HMM-GARCH-EVT VaR Model', fontsize=20)

# Panel 1: The Final Hybrid VaR
ax1.plot(portfolio_returns, color='gray', alpha=0.3, label='Portfolio Returns')
ax1.plot(hybrid_df['VaR_GJR'], color='purple', linestyle=':', alpha=0.5, label='Component: GJR-GARCH VaR')
ax1.plot(hybrid_df['VaR_EVT'], color='red', linestyle=':', alpha=0.5, label='Component: EVT VaR')
ax1.plot(hybrid_df['VaR_Hybrid'], color='blue', linewidth=2.5, label='Final Hybrid VaR')
ax1.set_title('Final Blended VaR vs. Component Models', fontsize=16)
ax1.set_ylabel('Daily Return', fontsize=12)
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)

# Panel 2: The Stress Factor
hybrid_df['StressFactor'].plot(ax=ax2, color='teal', linewidth=2)
ax2.set_title('Market Stress Factor (Weight on GJR-GARCH Model)', fontsize=16)
ax2.set_ylabel('Stress Factor', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()



# --- 1. Define Custom, Tunable Weights and Blending Power ---

# --- YOU CAN CHANGE THESE PARAMETERS TO EXPERIMENT ---
nervous_weight = 0.3      # Weight on GJR-GARCH for the "Nervous" state
volatile_weight = 0.50     # Weight on GJR-GARCH for the "Volatile" state
blending_exponent = 2.0    # Set to 1.0 for linear, > 1.0 for stricter blending
# ---

# Get the regime indices sorted by volatility (calm to crisis)
volatilities = final_hmm_model.means_[:, 1]
regime_order = np.argsort(volatilities)
calm_idx, nervous_idx, volatile_idx, crisis_idx = regime_order

# Assign weights based on your rules
gjr_weights = np.zeros(4)
gjr_weights[nervous_idx] = nervous_weight
gjr_weights[volatile_idx] = volatile_weight
gjr_weights[crisis_idx] = 1.00 # Crisis state always gets full weight

print("--- Custom Tunable Regime Weights (for GJR-GARCH) ---")
print(f"Calm State Weight:    {gjr_weights[calm_idx]:.2f}")
print(f"Nervous State Weight: {gjr_weights[nervous_idx]:.2f}")
print(f"Volatile State Weight:{gjr_weights[volatile_idx]:.2f}")
print(f"Crisis State Weight:  {gjr_weights[crisis_idx]:.2f}")

# --- 2. Build the Hybrid Model with Non-Linear, Custom Blending ---
# Calculate the initial linear stress factor
custom_stress_factor = prob_df.dot(gjr_weights)

# Create the final DataFrame
hybrid_df = pd.DataFrame(index=custom_stress_factor.index)
hybrid_df['StressFactor_Linear'] = custom_stress_factor
hybrid_df['VaR_GJR'] = gjr_garch_var_99
hybrid_df['VaR_EVT'] = evt_var_99_series
hybrid_df.dropna(inplace=True)

# Apply the non-linear transformation (the exponent)
strict_stress_factor = hybrid_df['StressFactor_Linear']**blending_exponent
hybrid_df['StressFactor_Final'] = strict_stress_factor

# Apply the final blending formula
hybrid_df['VaR_Hybrid'] = (
    (1 - hybrid_df['StressFactor_Final']) * hybrid_df['VaR_EVT'] +
    hybrid_df['StressFactor_Final'] * hybrid_df['VaR_GJR']
)
print("\n✅ Final non-linear, custom hybrid VaR calculation complete.")


# --- 3. Visualization ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 12), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)
fig.suptitle(f'Final Hybrid VaR (Exponent = {blending_exponent})', fontsize=20)

ax1.plot(portfolio_returns, color='gray', alpha=0.3, label='Portfolio Returns')
ax1.plot(hybrid_df['VaR_GJR'], color='purple', linestyle=':', alpha=0.5, label='Component: GJR-GARCH VaR')
ax1.plot(hybrid_df['VaR_EVT'], color='red', linestyle=':', alpha=0.5, label='Component: EVT VaR')
ax1.plot(hybrid_df['VaR_Hybrid'], color='blue', linewidth=2.5, label='Final Hybrid VaR')
ax1.set_title('Final Blended VaR vs. Component Models', fontsize=16)
ax1.set_ylabel('Daily Return', fontsize=12)
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)

ax2.plot(hybrid_df['StressFactor_Linear'], color='gray', linestyle=':', alpha=0.7, label='Original Linear Factor')
ax2.plot(hybrid_df['StressFactor_Final'], color='teal', linewidth=2, label='Final Non-Linear Factor')
ax2.set_title('Market Stress Factor (Final Weight on GJR-GARCH Model)', fontsize=16)
ax2.set_ylabel('Stress Factor', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()




# --- 1. Define Custom, Tunable Weights for Each Regime ---

# --- YOU CAN CHANGE THESE WEIGHTS TO EXPERIMENT ---
nervous_weight = 0.10  # Weight on GJR-GARCH for the "Nervous" state
volatile_weight = 0.50 # Weight on GJR-GARCH for the "Volatile" state
# ---

# Get the regime indices sorted by volatility (calm to crisis)
volatilities = final_hmm_model.means_[:, 1]
regime_order = np.argsort(volatilities)
calm_idx, nervous_idx, volatile_idx, crisis_idx = regime_order

# Assign weights based on your rules
gjr_weights = np.zeros(4)
gjr_weights[nervous_idx] = nervous_weight
gjr_weights[volatile_idx] = volatile_weight
gjr_weights[crisis_idx] = 1.00 # Crisis state always gets full weight

print("--- Custom Tunable Regime Weights (for GJR-GARCH) ---")
print(f"Calm State Weight:    {gjr_weights[calm_idx]:.2f}")
print(f"Nervous State Weight: {gjr_weights[nervous_idx]:.2f}")
print(f"Volatile State Weight:{gjr_weights[volatile_idx]:.2f}")
print(f"Crisis State Weight:  {gjr_weights[crisis_idx]:.2f}")

# --- 2. Build the Hybrid Model with Custom Linear Blending ---
# Calculate the final daily stress factor by dotting the daily probabilities
# with our custom weights vector
custom_stress_factor = prob_df.dot(gjr_weights)
custom_stress_factor.name = "StressFactor_Custom"

# Create the final DataFrame
hybrid_df = pd.DataFrame(index=custom_stress_factor.index)
hybrid_df['StressFactor_Custom'] = custom_stress_factor
hybrid_df['VaR_GJR'] = gjr_garch_var_99
hybrid_df['VaR_EVT'] = evt_var_9d9_series
hybrid_df.dropna(inplace=True)

# Apply the linear blending formula
hybrid_df['VaR_Hybrid'] = (
    (1 - hybrid_df['StressFactor_Custom']) * hybrid_df['VaR_EVT'] +
    hybrid_df['StressFactor_Custom'] * hybrid_df['VaR_GJR']
)
print("\n✅ Final custom hybrid VaR calculation complete.")


# --- 3. Visualization ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 12), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)
fig.suptitle('Final Hybrid VaR with Custom Linear Blending', fontsize=20)

ax1.plot(portfolio_returns, color='gray', alpha=0.3, label='Portfolio Returns')
ax1.plot(hybrid_df['VaR_GJR'], color='purple', linestyle=':', alpha=0.5, label='Component: GJR-GARCH VaR')
ax1.plot(hybrid_df['VaR_EVT'], color='red', linestyle=':', alpha=0.5, label='Component: EVT VaR')
ax1.plot(hybrid_df['VaR_Hybrid'], color='blue', linewidth=2.5, label='Final Hybrid VaR')
ax1.set_title('Final Blended VaR vs. Component Models', fontsize=16)
ax1.set_ylabel('Daily Return', fontsize=12)
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)

hybrid_df['StressFactor_Custom'].plot(ax=ax2, color='teal', linewidth=2)
ax2.set_title('Market Stress Factor (Custom Weight on GJR-GARCH Model)', fontsize=16)
ax2.set_ylabel('Stress Factor', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()