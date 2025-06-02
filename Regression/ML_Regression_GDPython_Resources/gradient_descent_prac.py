import pandas as pd
import numpy as np



def gradient_descent(x,y,lr=0.1, epochs=3000):

    x_min,x_max, = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_scaled = (x-x_min)/(x_max- x_min)
    y_scaled = (y-y_min)/(y_max- y_min)

    # Initialize parameters
    m = 0.0
    b = 0.0


    for epoch in range(epochs):
        y_pred = m*x_scaled + b
        error = y_scaled-y_pred

        cost = np.mean(error ** 2)   # Mean squared error

        # Calculate gradients
        db = -2 * np.mean(error)  # Derivative w.r.t. intercept b
        dm = -2 * np.mean(error * x_scaled)

        m -= lr * dm
        b -= lr * db
        # Optional: Print cost every 100 iterations to monitor progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}, b = {b}, m = {m}")

    # Scale back the coefficients to original scale, this comes from soving equation and comparing y = mx + b.
    m_original = m * (y_max - y_min) / (x_max - x_min)
    b_original = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)

    return b_original, m_original
    


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Devlopment_Learn\test\Regression\ML_Regression_GDPython_Resources\home_prices.csv")
    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()
    b, m = gradient_descent(x, y)
    print(f"Final Results: m={m}, b={b}")
