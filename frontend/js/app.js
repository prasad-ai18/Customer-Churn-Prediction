/**
 * Customer Churn Intelligence & Prediction Platform - Application Logic
 * Integrates with FastAPI endpoints (/predict, /model-info, /metrics, /customers, /health)
 */

const API_BASE = ""; // Relative path when served via FastAPI StaticFiles

// State
let appState = {
  activeTab: "dashboard",
  modelInfo: null,
  metrics: null,
  customers: [],
  filteredCustomers: []
};

// Presets for quick demonstration
const PRESETS = {
  high_risk: {
    customerID: "CUST-HIGH-RISK",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 2,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 98.50,
    TotalCharges: 197.00
  },
  low_risk: {
    customerID: "CUST-LOYAL-RET",
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "Yes",
    Dependents: "Yes",
    tenure: 64,
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Two year",
    PaperlessBilling: "No",
    PaymentMethod: "Credit card (automatic)",
    MonthlyCharges: 68.20,
    TotalCharges: 4364.80
  },
  medium_risk: {
    customerID: "CUST-BORDERLINE",
    gender: "Female",
    SeniorCitizen: 1,
    Partner: "No",
    Dependents: "No",
    tenure: 14,
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "Yes",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "One year",
    PaperlessBilling: "Yes",
    PaymentMethod: "Bank transfer (automatic)",
    MonthlyCharges: 79.90,
    TotalCharges: 1118.60
  }
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initNavigation();
  initFormHandlers();
  checkAuth();
});

// Theme Management (Bright & Dark Mode)
window.toggleTheme = function() {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  const next = (current === "light") ? "dark" : "light";
  applyTheme(next);
};

function initTheme() {
  const saved = localStorage.getItem("churn_theme") || "light";
  applyTheme(saved);
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("churn_theme", theme);

  const sunIcon = document.getElementById("theme-icon-sun");
  const moonIcon = document.getElementById("theme-icon-moon");
  const toggleBtn = document.getElementById("theme-toggle");

  if (theme === "dark") {
    if (sunIcon) sunIcon.style.display = "block";
    if (moonIcon) moonIcon.style.display = "none";
    if (toggleBtn) toggleBtn.setAttribute("title", "Switch to Bright Mode");
  } else {
    if (sunIcon) sunIcon.style.display = "none";
    if (moonIcon) moonIcon.style.display = "block";
    if (toggleBtn) toggleBtn.setAttribute("title", "Switch to Dark Mode");
  }

  // If currently on performance tab, re-render curves with new theme palette
  if (appState.activeTab === "performance" && appState.metrics) {
    renderCharts();
  }
}

// Authentication & Demo Users
const DEMO_USERS = {
  alex: {
    email: "alex.turner@telecom.ai",
    name: "Alex Turner",
    role: "Retention Lead",
    avatar: "AT"
  },
  sarah: {
    email: "sarah.chen@telecom.ai",
    name: "Sarah Chen",
    role: "Data Scientist",
    avatar: "SC"
  }
};

window.fillDemoLogin = function(userKey) {
  const u = DEMO_USERS[userKey];
  if (!u) return;
  const emailInput = document.getElementById("login-email");
  const pwdInput = document.getElementById("login-password");
  if (emailInput) emailInput.value = u.email;
  if (pwdInput) pwdInput.value = "demo2026";
  showToast(`Loaded ${u.name} credentials`);
};

window.handleLoginSubmit = function(event) {
  if (event) event.preventDefault();

  const emailInput = document.getElementById("login-email");
  const email = (emailInput?.value || "").trim().toLowerCase();

  let user = Object.values(DEMO_USERS).find(u => u.email.toLowerCase() === email);
  if (!user) {
    user = {
      email: email || "user@telecom.ai",
      name: email ? email.split("@")[0].replace(".", " ").replace(/\b\w/g, l => l.toUpperCase()) : "Guest Analyst",
      role: "Operations Analyst",
      avatar: email ? email.substring(0, 2).toUpperCase() : "GA"
    };
  }

  const btn = document.getElementById("btn-login-submit");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Signing in...`;
  }

  setTimeout(() => {
    localStorage.setItem("churn_auth_user", JSON.stringify(user));
    unlockDashboard(user);
    showToast(`Welcome back, ${user.name}!`);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path><polyline points="10 17 15 12 10 7"></polyline><line x1="15" y1="12" x2="3" y2="12"></line></svg> Sign In to Platform`;
    }
  }, 350);
};

window.handleSignOut = function() {
  localStorage.removeItem("churn_auth_user");
  const appView = document.getElementById("app-view");
  const loginView = document.getElementById("login-view");
  if (appView) appView.style.display = "none";
  if (loginView) loginView.style.display = "flex";
  showToast("You have been signed out.");
};

function checkAuth() {
  const saved = localStorage.getItem("churn_auth_user");
  if (saved) {
    try {
      const user = JSON.parse(saved);
      unlockDashboard(user);
      return true;
    } catch (e) {
      localStorage.removeItem("churn_auth_user");
    }
  }
  // Show login view by default
  const appView = document.getElementById("app-view");
  const loginView = document.getElementById("login-view");
  if (appView) appView.style.display = "none";
  if (loginView) loginView.style.display = "flex";
  return false;
}

function unlockDashboard(user) {
  const appView = document.getElementById("app-view");
  const loginView = document.getElementById("login-view");
  if (loginView) loginView.style.display = "none";
  if (appView) appView.style.display = "flex";

  const nameEl = document.getElementById("header-user-name");
  const roleEl = document.getElementById("header-user-role");
  const avatarEl = document.getElementById("header-user-avatar");
  if (nameEl) nameEl.textContent = user.name;
  if (roleEl) roleEl.textContent = user.role;
  if (avatarEl) avatarEl.textContent = user.avatar;

  loadPlatformData();
}

// Tab Navigation
function initNavigation() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  appState.activeTab = tabId;
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

  const targetBtn = document.querySelector(`.nav-btn[data-tab="${tabId}"]`);
  const targetPane = document.getElementById(tabId);

  if (targetBtn) targetBtn.classList.add("active");
  if (targetPane) targetPane.classList.add("active");

  // Re-render charts when switching to performance tab
  if (tabId === "performance" && appState.metrics) {
    renderCharts();
  }

  // Refresh explanation tab when switching to explanation
  if (tabId === "explanation" && appState.latestPrediction) {
    renderExplanationTab(appState.latestPrediction, appState.latestCustomer);
  }
}

// Fetch Initial Platform Metadata, Metrics & Customers
async function loadPlatformData() {
  try {
    // 1. Health check
    const healthRes = await fetch(`${API_BASE}/health`);
    const health = await healthRes.json();
    const statusText = document.getElementById("system-status-text");
    if (statusText) statusText.textContent = `Online • ${health.model_name || "Ready"}`;

    // 2. Model Info
    const infoRes = await fetch(`${API_BASE}/model-info`);
    if (infoRes.ok) {
      appState.modelInfo = await infoRes.json();
      populateDashboardMetrics(appState.modelInfo);
    }

    // 3. Model Comparison Metrics
    const metricsRes = await fetch(`${API_BASE}/metrics`);
    if (metricsRes.ok) {
      appState.metrics = await metricsRes.json();
      populatePerformanceTab(appState.metrics);
    }

    // 4. Sample Customers for Risk Explorer
    const custRes = await fetch(`${API_BASE}/customers?limit=60`);
    if (custRes.ok) {
      const data = await custRes.json();
      appState.customers = data.customers || [];
      appState.filteredCustomers = [...appState.customers];
      renderCustomerTable();
      populateDashboardAggregates(appState.customers);
    }

    // 5. Initial real prediction run for simulator & explanation
    await runPrediction();

  } catch (err) {
    console.error("Error loading platform data:", err);
    showToast("Failed to connect to backend API: " + err.message);
  }
}

// Dashboard Summary Cards
function populateDashboardMetrics(info) {
  const m = info.primary_metrics || {};
  const totalSamples = (info.train_samples || 0) + (info.val_samples || 0) + (info.test_samples || 0);

  const elTotal = document.getElementById("kpi-total-dataset");
  if (elTotal) elTotal.textContent = totalSamples ? totalSamples.toLocaleString() : "7,043";

  const elRoc = document.getElementById("kpi-roc-auc");
  if (elRoc) elRoc.textContent = m.roc_auc ? (m.roc_auc * 100).toFixed(1) + "%" : "84.6%";

  const elF1 = document.getElementById("kpi-f1-score");
  if (elF1) elF1.textContent = m.f1_score ? (m.f1_score * 100).toFixed(1) + "%" : "62.7%";

  const elModel = document.getElementById("kpi-champion-model");
  if (elModel) elModel.textContent = formatModelName(info.model_name);

  // Render global feature importance on dashboard panel
  if (info.global_feature_importance) {
    renderGlobalImportance(info.global_feature_importance);
  }
}

function populateDashboardAggregates(customers) {
  if (!customers.length) return;
  const total = customers.length;
  let churnSum = 0;
  let highRiskCount = 0;

  customers.forEach(c => {
    churnSum += c.predicted_churn_prob || 0;
    if (c.risk_level === "High") highRiskCount++;
  });

  const avgProb = churnSum / total;
  const churnRatePct = (avgProb * 100).toFixed(1);

  const elAvgProb = document.getElementById("kpi-avg-churn");
  if (elAvgProb) elAvgProb.textContent = churnRatePct + "%";

  const elHighRisk = document.getElementById("kpi-high-risk-count");
  if (elHighRisk) elHighRisk.textContent = `${highRiskCount} (${((highRiskCount / total) * 100).toFixed(0)}%)`;
}

// Form Handlers & Prediction Flow
function initFormHandlers() {
  const form = document.getElementById("prediction-form");
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      await runPrediction();
    });
  }

  // Preset Buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      if (PRESETS[presetKey]) {
        fillForm(PRESETS[presetKey]);
        runPrediction();
      }
    });
  });

  // Risk Explorer Table Search & Filter
  const searchInput = document.getElementById("customer-search");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      filterCustomers(e.target.value, document.getElementById("risk-filter")?.value);
    });
  }

  const riskFilter = document.getElementById("risk-filter");
  if (riskFilter) {
    riskFilter.addEventListener("change", (e) => {
      filterCustomers(document.getElementById("customer-search")?.value, e.target.value);
    });
  }
}

function fillForm(data) {
  for (const [key, value] of Object.entries(data)) {
    const field = document.getElementById(`field-${key}`);
    if (field) {
      field.value = value;
    }
  }
}

async function runPrediction() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  const submitBtn = document.getElementById("btn-predict");
  const originalBtnText = submitBtn.innerHTML;
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="spinner"></span> Analyzing Risk Drivers...`;

  const payload = {
    customerID: document.getElementById("field-customerID")?.value || "CUST-SIM",
    gender: document.getElementById("field-gender")?.value,
    SeniorCitizen: parseInt(document.getElementById("field-SeniorCitizen")?.value || "0"),
    Partner: document.getElementById("field-Partner")?.value,
    Dependents: document.getElementById("field-Dependents")?.value,
    tenure: parseInt(document.getElementById("field-tenure")?.value || "1"),
    PhoneService: document.getElementById("field-PhoneService")?.value,
    MultipleLines: document.getElementById("field-MultipleLines")?.value,
    InternetService: document.getElementById("field-InternetService")?.value,
    OnlineSecurity: document.getElementById("field-OnlineSecurity")?.value,
    OnlineBackup: document.getElementById("field-OnlineBackup")?.value,
    DeviceProtection: document.getElementById("field-DeviceProtection")?.value,
    TechSupport: document.getElementById("field-TechSupport")?.value,
    StreamingTV: document.getElementById("field-StreamingTV")?.value,
    StreamingMovies: document.getElementById("field-StreamingMovies")?.value,
    Contract: document.getElementById("field-Contract")?.value,
    PaperlessBilling: document.getElementById("field-PaperlessBilling")?.value,
    PaymentMethod: document.getElementById("field-PaymentMethod")?.value,
    MonthlyCharges: parseFloat(document.getElementById("field-MonthlyCharges")?.value || "0"),
    TotalCharges: parseFloat(document.getElementById("field-TotalCharges")?.value || "0")
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction request failed.");
    }

    const result = await res.json();
    appState.latestPrediction = result;
    appState.latestCustomer = payload;
    renderPredictionResult(result);
    renderExplanationTab(result, payload);
  } catch (err) {
    console.error("Prediction error:", err);
    showToast("Error running prediction: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalBtnText;
  }
}

function renderPredictionResult(res) {
  const prob = res.churn_probability;
  const pct = Math.round(prob * 100);
  const risk = res.risk_level;

  // Animate Gauge
  const probText = document.getElementById("gauge-prob-val");
  if (probText) probText.textContent = `${pct}%`;

  const circleBar = document.getElementById("gauge-bar");
  if (circleBar) {
    const maxOffset = 440;
    const offset = maxOffset - (maxOffset * prob);
    circleBar.style.strokeDashoffset = offset;

    if (risk === "High") {
      circleBar.style.stroke = "var(--color-high-risk)";
    } else if (risk === "Medium") {
      circleBar.style.stroke = "var(--color-medium-risk)";
    } else {
      circleBar.style.stroke = "var(--color-low-risk)";
    }
  }

  // Risk Badge
  const badge = document.getElementById("result-risk-badge");
  if (badge) {
    badge.className = `risk-badge ${risk.toLowerCase()}`;
    badge.innerHTML = `<span>●</span> ${risk} Churn Risk (${(prob * 100).toFixed(1)}%)`;
  }

  // Recommendation text based on top driver
  const recEl = document.getElementById("retention-action-text");
  if (recEl) {
    const topRisk = res.explanation.top_risk_drivers[0];
    if (risk === "High") {
      recEl.innerHTML = `<strong>Priority Action:</strong> High risk primarily driven by <em>${topRisk ? topRisk.title : "Contract Term"}</em>. Recommend offering a discounted 1-year contract extension or tech support bundle.`;
      recEl.style.display = "block";
    } else if (risk === "Medium") {
      recEl.innerHTML = `<strong>Monitor:</strong> Moderate churn signal. Customer exhibits sensitivity to <em>${topRisk ? topRisk.title : "pricing/support"}</em>. Consider proactive satisfaction check.`;
      recEl.style.display = "block";
    } else {
      recEl.innerHTML = `<strong>Healthy Account:</strong> Strong retention drivers present. Maintain standard engagement.`;
      recEl.style.display = "block";
    }
  }

  // SHAP Drivers Lists
  const riskContainer = document.getElementById("risk-drivers-list");
  if (riskContainer) {
    riskContainer.innerHTML = "";
    res.explanation.top_risk_drivers.forEach(d => {
      const item = document.createElement("div");
      item.className = "driver-item risk";
      item.innerHTML = `
        <span>${d.title}</span>
        <span class="driver-impact risk">+${d.impact.toFixed(3)}</span>
      `;
      riskContainer.appendChild(item);
    });
  }

  const retContainer = document.getElementById("retention-drivers-list");
  if (retContainer) {
    retContainer.innerHTML = "";
    res.explanation.top_retention_drivers.forEach(d => {
      const item = document.createElement("div");
      item.className = "driver-item retention";
      item.innerHTML = `
        <span>${d.title}</span>
        <span class="driver-impact retention">${d.impact.toFixed(3)}</span>
      `;
      retContainer.appendChild(item);
    });
  }
}

// Risk Explorer Filtering & Rendering
function filterCustomers(searchVal = "", riskVal = "all") {
  const query = (searchVal || "").toLowerCase().trim();
  const risk = (riskVal || "all").toLowerCase();

  appState.filteredCustomers = appState.customers.filter(c => {
    const matchSearch = !query || 
      (c.customerID && c.customerID.toLowerCase().includes(query)) ||
      (c.Contract && c.Contract.toLowerCase().includes(query)) ||
      (c.PaymentMethod && c.PaymentMethod.toLowerCase().includes(query));

    const matchRisk = risk === "all" || (c.risk_level && c.risk_level.toLowerCase() === risk);
    return matchSearch && matchRisk;
  });

  renderCustomerTable();
}

function renderCustomerTable() {
  const tbody = document.getElementById("customer-table-body");
  if (!tbody) return;

  tbody.innerHTML = "";
  if (!appState.filteredCustomers.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 2rem; color: var(--text-dim);">No customer records matched your query.</td></tr>`;
    return;
  }

  appState.filteredCustomers.forEach(c => {
    const tr = document.createElement("tr");
    const riskClass = (c.risk_level || "low").toLowerCase();
    const probPct = ((c.predicted_churn_prob || 0) * 100).toFixed(1);

    tr.innerHTML = `
      <td><strong>${c.customerID}</strong></td>
      <td><span class="risk-badge ${riskClass}" style="margin:0; padding:0.2rem 0.6rem; font-size:0.75rem;">${c.risk_level} (${probPct}%)</span></td>
      <td>${c.tenure} mos</td>
      <td>${c.Contract}</td>
      <td>${c.InternetService}</td>
      <td>$${parseFloat(c.MonthlyCharges || 0).toFixed(2)}</td>
      <td>${c.actual_churn === 1 ? '<span style="color:var(--color-high-risk); font-weight:700;">Churned</span>' : '<span style="color:var(--color-low-risk);">Retained</span>'}</td>
      <td><button class="btn-inspect" onclick="inspectCustomer('${c.customerID}')">Simulate</button></td>
    `;
    tbody.appendChild(tr);
  });
}

window.inspectCustomer = function(customerID) {
  const cust = appState.customers.find(c => c.customerID === customerID);
  if (!cust) return;

  fillForm(cust);
  switchTab("prediction");
  runPrediction();
  showToast(`Loaded customer ${customerID} into Prediction Simulator`);
};

// Render Detailed SHAP Explanation Tab
function renderExplanationTab(res, custData) {
  if (!res || !res.explanation) return;

  const idEl = document.getElementById("expl-cust-id");
  if (idEl) idEl.textContent = custData?.customerID || "CUST-SIMULATED";

  const subEl = document.getElementById("expl-cust-subtitle");
  if (subEl && custData) {
    const monthlyFormatted = typeof custData.MonthlyCharges === "number" ? custData.MonthlyCharges.toFixed(2) : custData.MonthlyCharges;
    subEl.textContent = `Tenure: ${custData.tenure} mos • Contract: ${custData.Contract} • Internet: ${custData.InternetService} • Monthly: $${monthlyFormatted}`;
  }

  const probEl = document.getElementById("expl-cust-prob");
  if (probEl) probEl.textContent = `${(res.churn_probability * 100).toFixed(1)}%`;

  const badgeEl = document.getElementById("expl-cust-badge");
  if (badgeEl) {
    badgeEl.className = `risk-badge ${res.risk_level.toLowerCase()}`;
    badgeEl.textContent = `${res.risk_level} Risk`;
  }

  const container = document.getElementById("shap-waterfall-container");
  if (!container) return;
  container.innerHTML = "";

  const riskDrivers = res.explanation.top_risk_drivers || [];
  const retDrivers = res.explanation.top_retention_drivers || [];

  // Determine max absolute impact to normalize bar widths
  let maxImpact = 0.01;
  riskDrivers.forEach(d => { if (Math.abs(d.impact) > maxImpact) maxImpact = Math.abs(d.impact); });
  retDrivers.forEach(d => { if (Math.abs(d.impact) > maxImpact) maxImpact = Math.abs(d.impact); });

  // Render positive risk drivers (pushing churn probability up)
  riskDrivers.forEach(d => {
    const row = document.createElement("div");
    row.className = "waterfall-row";
    const barPct = Math.min(100, Math.max(14, (Math.abs(d.impact) / maxImpact) * 100));
    row.innerHTML = `
      <div class="waterfall-label" title="${d.title}">${d.title}</div>
      <div class="waterfall-track">
        <div class="waterfall-bar risk" style="width: ${barPct}%;">+${d.impact.toFixed(3)}</div>
      </div>
      <div class="waterfall-val risk">+${d.impact.toFixed(3)}</div>
    `;
    container.appendChild(row);
  });

  // Render negative retention drivers (pushing churn probability down)
  retDrivers.forEach(d => {
    const row = document.createElement("div");
    row.className = "waterfall-row";
    const barPct = Math.min(100, Math.max(14, (Math.abs(d.impact) / maxImpact) * 100));
    row.innerHTML = `
      <div class="waterfall-label" title="${d.title}">${d.title}</div>
      <div class="waterfall-track">
        <div class="waterfall-bar retention" style="width: ${barPct}%;">${d.impact.toFixed(3)}</div>
      </div>
      <div class="waterfall-val retention">${d.impact.toFixed(3)}</div>
    `;
    container.appendChild(row);
  });
}

// Interactive Counterfactual Simulator
window.applyCounterfactual = async function(type) {
  let changeDesc = "";
  if (type === "contract_1yr") {
    const el = document.getElementById("field-Contract");
    if (el) el.value = "One year";
    changeDesc = "Contract upgraded to 1-Year";
  } else if (type === "add_tech_support") {
    const el = document.getElementById("field-TechSupport");
    if (el) el.value = "Yes";
    changeDesc = "Tech Support added";
  } else if (type === "add_security") {
    const el = document.getElementById("field-OnlineSecurity");
    if (el) el.value = "Yes";
    changeDesc = "Online Security suite added";
  } else if (type === "switch_autopay") {
    const el = document.getElementById("field-PaymentMethod");
    if (el) el.value = "Bank transfer (automatic)";
    changeDesc = "Payment converted to Auto-Bank Transfer";
  }

  showToast(`Simulating: ${changeDesc}...`);
  await runPrediction();
  showToast(`Applied ${changeDesc}! Churn risk recalculated.`);
};

// Model Performance Tab
function populatePerformanceTab(comp) {
  const grid = document.getElementById("models-compare-grid");
  if (!grid) return;

  grid.innerHTML = "";
  const champion = appState.modelInfo?.model_name || "logistic_regression_baseline";

  for (const [name, m] of Object.entries(comp)) {
    const isChamp = (name === champion);
    const card = document.createElement("div");
    card.className = "panel-card";
    card.innerHTML = `
      <div class="panel-title">
        <span>${formatModelName(name)}</span>
        <span class="model-card-badge ${isChamp ? 'champion' : ''}">${isChamp ? '★ Champion' : 'Candidate'}</span>
      </div>
      <div class="model-stat-row"><span class="stat-label">ROC-AUC:</span><span class="stat-val">${(m.roc_auc * 100).toFixed(2)}%</span></div>
      <div class="model-stat-row"><span class="stat-label">PR-AUC (Avg Precision):</span><span class="stat-val">${(m.pr_auc * 100).toFixed(2)}%</span></div>
      <div class="model-stat-row"><span class="stat-label">F1-Score:</span><span class="stat-val">${(m.f1_score * 100).toFixed(2)}%</span></div>
      <div class="model-stat-row"><span class="stat-label">Recall (Sensitivity):</span><span class="stat-val">${(m.recall * 100).toFixed(2)}%</span></div>
      <div class="model-stat-row"><span class="stat-label">Precision:</span><span class="stat-val">${(m.precision * 100).toFixed(2)}%</span></div>
      <div class="model-stat-row"><span class="stat-label">Accuracy:</span><span class="stat-val">${(m.accuracy * 100).toFixed(2)}%</span></div>
    `;
    grid.appendChild(card);
  }

  // Populate Confusion Matrix for Champion Model
  const champMetrics = comp[champion];
  if (champMetrics && champMetrics.confusion_matrix) {
    const cm = champMetrics.confusion_matrix;
    document.getElementById("cm-tp").textContent = cm.true_positive || 0;
    document.getElementById("cm-fp").textContent = cm.false_positive || 0;
    document.getElementById("cm-tn").textContent = cm.true_negative || 0;
    document.getElementById("cm-fn").textContent = cm.false_negative || 0;
  }

  renderCharts();
}

// Canvas Visualizations
function renderCharts() {
  if (!appState.metrics) return;
  const champion = appState.modelInfo?.model_name || "logistic_regression_baseline";
  const m = appState.metrics[champion];
  if (!m) return;

  // Draw ROC Curve
  if (m.roc_curve) {
    drawCurve(
      "roc-canvas",
      m.roc_curve.fpr,
      m.roc_curve.tpr,
      "False Positive Rate",
      "True Positive Rate",
      "#0284c7",
      true
    );
  }

  // Draw PR Curve
  if (m.pr_curve) {
    drawCurve(
      "pr-canvas",
      m.pr_curve.recall,
      m.pr_curve.precision,
      "Recall",
      "Precision",
      "#10b981",
      false
    );
  }
}

function drawCurve(canvasId, xVals, yVals, xLabel, yLabel, color, isRoc = false) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !xVals || !yVals) return;

  const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  const isLight = currentTheme === "light";

  const ctx = canvas.getContext("2d");
  const width = canvas.width = canvas.parentElement.clientWidth;
  const height = canvas.height = canvas.parentElement.clientHeight;

  ctx.clearRect(0, 0, width, height);

  const padLeft = 45;
  const padBottom = 35;
  const padTop = 15;
  const padRight = 15;

  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  // Draw Grid Lines
  ctx.strokeStyle = isLight ? "rgba(14, 165, 233, 0.12)" : "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (plotH / 4) * i;
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);

    const x = padLeft + (plotW / 4) * i;
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, height - padBottom);
  }
  ctx.stroke();

  // If ROC, draw baseline diagonal (random classifier = 0.5)
  if (isRoc) {
    ctx.strokeStyle = isLight ? "rgba(14, 165, 233, 0.35)" : "rgba(255, 255, 255, 0.25)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padLeft, height - padBottom);
    ctx.lineTo(width - padRight, padTop);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Plot Curve
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();

  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + xVals[i] * plotW;
    const y = padTop + (1.0 - yVals[i]) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw Axis Labels
  ctx.fillStyle = isLight ? "#475569" : "#94a3b8";
  ctx.font = "10px Inter, sans-serif";
  ctx.fillText("0.0", padLeft - 20, height - padBottom + 4);
  ctx.fillText("1.0", padLeft - 20, padTop + 8);
  ctx.fillText("1.0", width - padRight - 15, height - padBottom + 16);
  ctx.fillText(xLabel, width / 2 - 30, height - 8);
}

function renderGlobalImportance(importances) {
  const container = document.getElementById("global-importance-list");
  if (!container) return;

  container.innerHTML = "";
  const maxImp = Math.max(...importances.map(i => i.importance), 0.01);

  importances.slice(0, 8).forEach(item => {
    const barPct = Math.round((item.importance / maxImp) * 100);
    const row = document.createElement("div");
    row.style.marginBottom = "0.75rem";
    row.innerHTML = `
      <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:0.25rem;">
        <span style="color:var(--text-main);">${item.title}</span>
        <span style="font-weight:700; color:var(--accent-cyan); font-family:monospace;">${(item.importance * 100).toFixed(1)}%</span>
      </div>
      <div style="width:100%; height:6px; background:rgba(255,255,255,0.06); border-radius:9999px; overflow:hidden;">
        <div style="width:${barPct}%; height:100%; background:var(--accent-gradient); border-radius:9999px;"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function formatModelName(name) {
  if (!name) return "XGBoost";
  const names = {
    "logistic_regression_baseline": "Logistic Regression Baseline",
    "random_forest": "Random Forest Ensemble",
    "xgboost": "XGBoost Gradient Boosting"
  };
  return names[name] || name.replace(/_/g, " ").toUpperCase();
}

function showToast(msg) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = msg;
  toast.style.display = "block";
  setTimeout(() => {
    toast.style.display = "none";
  }, 4000);
}
