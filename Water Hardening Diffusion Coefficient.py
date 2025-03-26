import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

# Define the differential equation
def dc_dx(x, c, kc, a, vL):
    return (kc * a / vL) * (1 - c)

# Function to compute concentration profile and column height
def compute_and_plot(vL, dp, DL):
    # Define constants
    e = 0.43
    phi = 0.8
    muL = 1.0023e-6
    
    # Compute dimensionless numbers
    Re = (phi * dp * vL) / (e * muL)
    Sc = muL / DL
    
    # Calculate mass transfer coefficient
    kc = (1 + 1.5 * (1 - e)) * (DL / dp) * (2 + 0.644 * (np.sqrt(Re)) * (Sc ** (1/3)))
    a = (6 * (1 - e)) / (dp * phi)
    
    c0 = 0  # Initial concentration
    cf = 0.99  # Final concentration threshold
    
    # Function to determine column height L
    def find_L():
        def objective(L):
            sol = solve_ivp(dc_dx, [0, L], [c0], args=(kc, a, vL), dense_output=True)
            return sol.y[0, -1] - cf
        from scipy.optimize import root_scalar
        result = root_scalar(objective, bracket=[0.001, 10], method='brentq')
        return result.root
    
    L = find_L()
    x_vals = np.linspace(0, L, 100)
    sol = solve_ivp(dc_dx, [0, L], [c0], args=(kc, a, vL), t_eval=x_vals)
    
    return x_vals, sol.y[0] * 100, L  # Return x-values, concentration values, and column height

# Initialize default values
vL_init = 12/3600
dp_init = 0.01
DL_init = 8.52e-10

# Compute initial concentration profile
x_vals, c_vals, L = compute_and_plot(vL_init, dp_init, DL_init)

# Create the figure and plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.5)

# Plot initial concentration profile
line, = ax.plot(x_vals, c_vals, label='Percentage Saturation')
ax.set_xlabel('Column Height (m)')
ax.set_ylabel('Percentage Saturation (%)')
ax.set_title('Concentration Profile along Column Height')
ax.legend()
ax.grid()

# Define slider axes
axcolor = 'lightgoldenrodyellow'
ax_vL = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_dp = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_DL = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_reset = plt.axes([0.8, 0.3, 0.1, 0.04])
ax_text = plt.axes([0.4, 0.35, 0.4, 0.04])

# Create sliders for vL, dp, and DL
s_vL = Slider(ax_vL, 'vL (m/s)', 0.0005, 0.01, valinit=vL_init, valstep=0.0001)
s_dp = Slider(ax_dp, 'dp (m)', 0.001, 0.05, valinit=dp_init, valstep=0.0001)
s_DL = Slider(ax_DL, 'DL (m^2/s)', 1e-11, 1e-8, valinit=DL_init, valstep=1e-10)

# Create reset button
btn_reset = Button(ax_reset, 'Reset')

# Create text box for setting initial values
txt_box = TextBox(ax_text, 'Set Initial vL, dp, DL (comma-separated)')

# Function to update plot when sliders change
def update(val):
    vL = s_vL.val
    dp = s_dp.val
    DL = s_DL.val
    x_vals, c_vals, L = compute_and_plot(vL, dp, DL)
    line.set_xdata(x_vals)
    line.set_ydata(c_vals)
    ax.set_xlim([0, L])
    ax.set_ylim([0, 100])
    fig.canvas.draw_idle()

# Function to reset sliders to initial values
def reset(event):
    s_vL.reset()
    s_dp.reset()
    s_DL.reset()

# Function to update initial values from text box input
def set_initial_values(text):
    try:
        values = [float(v.strip()) for v in text.split(',')]
        if len(values) == 3:
            s_vL.set_val(values[0])
            s_dp.set_val(values[1])
            s_DL.set_val(values[2])
            update(None)  # Force update after setting new values
    except ValueError:
        print("Invalid input. Please enter three numeric values separated by commas.")

# Connect sliders and buttons to their respective functions
s_vL.on_changed(update)
s_dp.on_changed(update)
s_DL.on_changed(update)
btn_reset.on_clicked(reset)
txt_box.on_submit(set_initial_values)

# Display the plot
plt.show()
