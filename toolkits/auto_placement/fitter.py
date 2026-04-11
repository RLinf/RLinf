# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import numpy as np
from scipy.optimize import curve_fit


class DataFitter:
    def __init__(self, profile_data: dict):
        self.profile_data = profile_data
        self.x_data = np.array(list(self.profile_data.keys()))
        self.y_data = np.array(list(self.profile_data.values()))

        self.fit_params = {}
        self.best_fit_type = None

        self._perform_fitting()

    def _power_law(self, x, a, b):
        """y = a * x^b"""
        return a * np.power(x, b)

    def _exponential(self, x, a, b):
        """y = a * exp(b * x)"""
        return a * np.exp(b * x)

    def _logarithmic(self, x, a, b):
        """y = a + b * ln(x)"""
        return a + b * np.log(x)

    def _polynomial(self, x, a, b, c):
        """y = a*x^2 + b*x + c"""
        return a * x**2 + b * x + c

    def _perform_fitting(self):
        fit_results = {}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    self._power_law,
                    self.x_data,
                    self.y_data,
                    p0=[0.1, 0.5],
                    maxfev=5000,
                )
            y_pred = self._power_law(self.x_data, *popt)
            r_squared = self._calculate_r_squared(self.y_data, y_pred)
            fit_results["power_law"] = {
                "params": popt,
                "r_squared": r_squared,
                "function": self._power_law,
            }
        except:  # noqa: E722
            pass

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    self._exponential,
                    self.x_data,
                    self.y_data,
                    p0=[0.1, 0.01],
                    maxfev=5000,
                )
            y_pred = self._exponential(self.x_data, *popt)
            r_squared = self._calculate_r_squared(self.y_data, y_pred)
            fit_results["exponential"] = {
                "params": popt,
                "r_squared": r_squared,
                "function": self._exponential,
            }
        except:  # noqa: E722
            pass

        try:
            if np.all(self.x_data > 0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, pcov = curve_fit(
                        self._logarithmic,
                        self.x_data,
                        self.y_data,
                        p0=[0.1, 0.1],
                        maxfev=5000,
                    )
                y_pred = self._logarithmic(self.x_data, *popt)
                r_squared = self._calculate_r_squared(self.y_data, y_pred)
                fit_results["logarithmic"] = {
                    "params": popt,
                    "r_squared": r_squared,
                    "function": self._logarithmic,
                }
        except:  # noqa: E722
            pass

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    self._polynomial,
                    self.x_data,
                    self.y_data,
                    p0=[0.001, 0.01, 0.1],
                    maxfev=5000,
                )
            y_pred = self._polynomial(self.x_data, *popt)
            r_squared = self._calculate_r_squared(self.y_data, y_pred)
            fit_results["polynomial"] = {
                "params": popt,
                "r_squared": r_squared,
                "function": self._polynomial,
            }
        except:  # noqa: E722
            pass

        assert fit_results is not None
        self.best_fit_type = max(
            fit_results.keys(), key=lambda x: fit_results[x]["r_squared"]
        )
        self.fit_params = fit_results[self.best_fit_type]

    def _calculate_r_squared(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def get_value(self, x):
        x = int(x)

        if x in self.profile_data:
            return self.profile_data[x]

        fit_function = self.fit_params["function"]
        params = self.fit_params["params"]

        if self.best_fit_type == "logarithmic" and x <= 0:
            raise ValueError(f"{x=} < 0 in logarithmic func")

        return float(fit_function(x, *params))

    def predict(self, x_values):
        if isinstance(x_values, (int, float)):
            x_values = [x_values]

        return [self.get_value(x) for x in x_values]

    def get_fit_info(self):
        return {
            "best_fit_type": self.best_fit_type,
            "r_squared": self.fit_params["r_squared"],
            "parameters": self.fit_params["params"],
            "profile_data": self.profile_data,
        }
