// ============================================================
// SCAMGRAPH SENTINEL - DASHBOARD
// Frontend Prediction Controller
// ============================================================

const API_URL = "http://127.0.0.1:5000";

document.addEventListener("DOMContentLoaded", () => {

    console.log("SCAMGRAPH SENTINEL dashboard loaded.");

    const form = document.getElementById("prediction-form");
    const resultBox = document.getElementById("prediction-result");
    const apiStatus = document.getElementById("api-status");

    if (!form) {
        console.error("ERROR: prediction-form not found.");
        return;
    }

    // --------------------------------------------------------
    // ELEMENTS
    // --------------------------------------------------------

    const amount = document.getElementById("amount");
    const txTime = document.getElementById("tx-time");
    const txDays = document.getElementById("tx-days");
    const hour = document.getElementById("hour");
    const day = document.getElementById("day");
    const dayMonth = document.getElementById("day-month");
    const month = document.getElementById("month");

    const customerTxCount =
        document.getElementById("customer-tx-count");

    const customerAvg =
        document.getElementById("customer-avg");

    const customerMax =
        document.getElementById("customer-max");

    const terminalTxCount =
        document.getElementById("terminal-tx-count");

    const terminalAvg =
        document.getElementById("terminal-avg");

    const weekend =
        document.getElementById("weekend");

    const night =
        document.getElementById("night");

    const highAmount =
        document.getElementById("high-amount");

    const analyzeButton =
        document.getElementById("predict-button");


    // --------------------------------------------------------
    // CHECK API
    // --------------------------------------------------------

    async function checkAPI() {

        try {

            const response =
                await fetch(`${API_URL}/health`);

            if (!response.ok) {
                throw new Error("API unavailable");
            }

            const data =
                await response.json();

            console.log("API health:", data);

            if (apiStatus) {
                apiStatus.textContent =
                    data.model_loaded
                        ? "ONLINE • MODEL ACTIVE"
                        : "ONLINE • MODEL ERROR";
            }

        } catch (error) {

            console.error("API health check failed:", error);

            if (apiStatus) {
                apiStatus.textContent =
                    "OFFLINE";
            }
        }
    }


    // --------------------------------------------------------
    // SAFE NUMBER
    // --------------------------------------------------------

    function numberValue(element, fallback = 0) {

        if (!element) {
            return fallback;
        }

        const value =
            parseFloat(element.value);

        return Number.isFinite(value)
            ? value
            : fallback;
    }


    // --------------------------------------------------------
    // CALCULATED FEATURES
    // --------------------------------------------------------

    function calculateFeatures() {

        const txAmount =
            numberValue(amount);

        const customerAverage =
            numberValue(customerAvg);

        const terminalAverage =
            numberValue(terminalAvg);

        const amountToCustomerAvg =
            customerAverage > 0
                ? txAmount / customerAverage
                : 0;

        const amountToTerminalAvg =
            terminalAverage > 0
                ? txAmount / terminalAverage
                : 0;

        return {
            amountToCustomerAvg,
            amountToTerminalAvg
        };
    }


    // --------------------------------------------------------
    // SHOW RESULT
    // --------------------------------------------------------

    function showResult(result) {

        if (!resultBox) {
            return;
        }

        const isFraud =
            Number(result.prediction) === 1;

        const percentage =
            Number(result.fraud_percentage || 0);

        const probability =
            Number(result.fraud_probability || 0);

        const score =
            Number(result.score || 0);

        resultBox.innerHTML = `

            <div class="prediction-result ${isFraud ? "fraud" : "legitimate"}">

                <div class="result-icon">
                    ${isFraud ? "⚠️" : "🛡️"}
                </div>

                <div class="result-title">
                    ${isFraud ? "FRAUD DETECTED" : "LEGITIMATE TRANSACTION"}
                </div>

                <div class="result-subtitle">
                    AI model analysis completed successfully
                </div>

                <div class="risk-meter">

                    <div class="risk-meter-header">
                        <span>Fraud Probability</span>
                        <strong>${percentage.toFixed(2)}%</strong>
                    </div>

                    <div class="risk-meter-track">

                        <div
                            class="risk-meter-fill"
                            style="width: ${Math.min(percentage, 100)}%"
                        ></div>

                    </div>

                </div>

                <div class="result-grid">

                    <div class="result-stat">
                        <span>Prediction</span>
                        <strong>${result.prediction}</strong>
                    </div>

                    <div class="result-stat">
                        <span>Probability</span>
                        <strong>${probability.toFixed(6)}</strong>
                    </div>

                    <div class="result-stat">
                        <span>Model Score</span>
                        <strong>${score.toFixed(6)}</strong>
                    </div>

                    <div class="result-stat">
                        <span>Threshold</span>
                        <strong>${Number(result.threshold || 0.5).toFixed(2)}</strong>
                    </div>

                </div>

                <div class="result-message">

                    ${
                        isFraud
                            ? "⚠️ This transaction shows behavior associated with fraudulent activity."
                            : "✓ No significant fraud signal was detected for this transaction."
                    }

                </div>

            </div>
        `;
    }


    // --------------------------------------------------------
    // SHOW ERROR
    // --------------------------------------------------------

    function showError(message) {

        if (!resultBox) {
            return;
        }

        resultBox.innerHTML = `

            <div class="prediction-error">

                <div class="result-icon">❌</div>

                <h3>Analysis Failed</h3>

                <p>${message}</p>

                <small>
                    Check that the Flask API is running on
                    ${API_URL}
                </small>

            </div>

        `;
    }


    // --------------------------------------------------------
    // FORM SUBMISSION
    // --------------------------------------------------------

    form.addEventListener("submit", async (event) => {

        event.preventDefault();

        console.log("Analyze button clicked.");

        if (analyzeButton) {
            analyzeButton.disabled = true;
            analyzeButton.textContent =
                "⏳ ANALYZING...";
        }

        try {

            const calculated =
                calculateFeatures();

            // ------------------------------------------------
            // BUILD EXACT 17 MODEL FEATURES
            // ------------------------------------------------

            const payload = {

                TX_AMOUNT:
                    numberValue(amount),

                TX_TIME_SECONDS:
                    numberValue(txTime),

                TX_TIME_DAYS:
                    numberValue(txDays),

                HOUR:
                    numberValue(hour),

                DAY_OF_WEEK:
                    numberValue(day),

                DAY_OF_MONTH:
                    numberValue(dayMonth),

                MONTH:
                    numberValue(month),

                IS_WEEKEND:
                    weekend && weekend.checked ? 1 : 0,

                IS_NIGHT:
                    night && night.checked ? 1 : 0,

                CUSTOMER_TX_COUNT:
                    numberValue(customerTxCount),

                CUSTOMER_AVG_AMOUNT:
                    numberValue(customerAvg),

                CUSTOMER_MAX_AMOUNT:
                    numberValue(customerMax),

                AMOUNT_TO_CUSTOMER_AVG:
                    calculated.amountToCustomerAvg,

                TERMINAL_TX_COUNT:
                    numberValue(terminalTxCount),

                TERMINAL_AVG_AMOUNT:
                    numberValue(terminalAvg),

                AMOUNT_TO_TERMINAL_AVG:
                    calculated.amountToTerminalAvg,

                HIGH_AMOUNT:
                    highAmount && highAmount.checked ? 1 : 0
            };


            console.log(
                "Sending prediction payload:",
                payload
            );


            // ------------------------------------------------
            // BASIC VALIDATION
            // ------------------------------------------------

            if (payload.TX_AMOUNT <= 0) {
                throw new Error(
                    "Transaction amount must be greater than 0."
                );
            }

            if (payload.CUSTOMER_AVG_AMOUNT <= 0) {
                throw new Error(
                    "Customer average amount must be greater than 0."
                );
            }

            if (payload.TERMINAL_AVG_AMOUNT <= 0) {
                throw new Error(
                    "Terminal average amount must be greater than 0."
                );
            }


            // ------------------------------------------------
            // SEND TO FLASK
            // ------------------------------------------------

            const response =
                await fetch(`${API_URL}/predict`, {

                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(payload)
                });


            const data =
                await response.json();

            console.log(
                "API prediction response:",
                data
            );


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction API returned an error."
                );
            }


            // ------------------------------------------------
            // DISPLAY RESULT
            // ------------------------------------------------

            showResult(data);

        } catch (error) {

            console.error(
                "Prediction error:",
                error
            );

            showError(
                error.message
            );

        } finally {

            if (analyzeButton) {

                analyzeButton.disabled = false;

                analyzeButton.textContent =
                    "🔍 ANALYZE TRANSACTION";
            }
        }

    });


    // --------------------------------------------------------
    // INITIALIZE
    // --------------------------------------------------------

    checkAPI();

});