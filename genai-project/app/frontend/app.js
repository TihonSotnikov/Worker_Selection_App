
const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsHost = window.location.host;
const wsBaseUrl = `${wsProtocol}//${wsHost}/api/ws`;

const state = {
  currentCandidate: null,
  currentPresetLabel: 'Кандидат',
};

const presetLabels = {
  green: 'Идеальный кандидат',
  red: 'Проблемный кандидат',
  yellow: 'Спорный кандидат',
  edge: 'Edge Case: требуется уточнение опыта',
};

const riskThemes = {
  LOW: {
    badge: 'low',
    bar: 'var(--success)',
    label: 'Низкий риск',
    short: 'Подходит',
    fallbackMargin: 0.05,
  },
  MEDIUM: {
    badge: 'medium',
    bar: 'var(--warning)',
    label: 'Нужна проверка',
    short: 'Проверить',
    fallbackMargin: 0.08,
  },
  HIGH: {
    badge: 'high',
    bar: 'var(--danger)',
    label: 'Высокий риск',
    short: 'Осторожно',
    fallbackMargin: 0.06,
  },
};

const detailLabels = {
  skills_verified_count: 'Подтверждённые навыки',
  years_experience: 'Опыт работы',
  age: 'Возраст',
  commute_time_minutes: 'Время в пути',
  shift_preference: 'График',
  salary_expectation: 'Ожидаемая зарплата',
  has_certifications: 'Сертификаты',
  education_level: 'Образование',
  previous_turnovers: 'Предыдущие увольнения',
  family_status: 'Семейный статус',
  housing_type: 'Тип жилья',
  has_transport: 'Личный транспорт',
};

const shiftLabels = {
  0: 'Дневная смена',
  1: 'Ночная смена',
  2: 'Любая смена',
};

const educationLabels = {
  0: 'Среднее',
  1: 'Среднее специальное',
  2: 'Колледж',
  3: 'Высшее',
};

const familyLabels = {
  0: 'Без семьи / не указано',
  1: 'В браке, без детей',
  2: 'В браке, есть дети',
  3: 'Один родитель',
};

const housingLabels = {
  0: 'Своё жильё',
  1: 'Аренда',
  2: 'Общежитие',
  3: 'С родителями',
};

const form = document.querySelector('#candidateForm');
const toast = document.querySelector('#toast');

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove('hidden');

  window.clearTimeout(showToast.timer);

  showToast.timer = window.setTimeout(() => {
    toast.classList.add('hidden');
  }, 5200);
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get('content-type') || '';

  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail =
      typeof payload === 'object' && payload.detail
        ? payload.detail
        : payload;

    throw new Error(detail || `HTTP ${response.status}`);
  }

  return payload;
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function currency(value) {
  return new Intl.NumberFormat('ru-RU').format(Number(value));
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, Number(value)));
}

function getRiskLevelByScore(score) {
  if (score >= 0.7) return 'LOW';
  if (score >= 0.4) return 'MEDIUM';
  return 'HIGH';
}

function getTheme(resultOrScore) {
  const level =
    typeof resultOrScore === 'number'
      ? getRiskLevelByScore(resultOrScore)
      : resultOrScore.risk_level || getRiskLevelByScore(resultOrScore.retention_probability);

  return riskThemes[level] || riskThemes.MEDIUM;
}

function getUncertainty(result, probability) {
  const theme = getTheme(result);

  const fallbackMargin = result.requires_review
    ? 0.12
    : theme.fallbackMargin;

  return {
    low: clamp(result.uncertainty_low ?? probability - fallbackMargin),
    high: clamp(result.uncertainty_high ?? probability + fallbackMargin),
    note:
      result.uncertainty_note ||
      'Ориентировочный коридор неопределённости. Это не финальный HR-вердикт.',
  };
}

function fillCandidateForm(candidate) {
  form.elements.skills_verified_count.value = candidate.skills_verified_count;
  form.elements.years_experience.value = candidate.years_experience;
  form.elements.age.value = candidate.age;
  form.elements.commute_time_minutes.value = candidate.commute_time_minutes;
  form.elements.shift_preference.value = String(candidate.shift_preference);
  form.elements.salary_expectation.value = candidate.salary_expectation;
  form.elements.has_certifications.checked = Boolean(candidate.has_certifications);
  form.elements.education_level.value = String(candidate.education_level);
  form.elements.previous_turnovers.value = candidate.previous_turnovers;
  form.elements.family_status.value = String(candidate.family_status);
  form.elements.housing_type.value = String(candidate.housing_type);
  form.elements.has_transport.checked = Boolean(candidate.has_transport);
}

function readCandidateForm() {
  return {
    skills_verified_count: Number(form.elements.skills_verified_count.value),
    years_experience: Number(form.elements.years_experience.value),
    age: Number(form.elements.age.value),
    commute_time_minutes: Number(form.elements.commute_time_minutes.value),
    shift_preference: Number(form.elements.shift_preference.value),
    salary_expectation: Number(form.elements.salary_expectation.value),
    has_certifications: Boolean(form.elements.has_certifications.checked),
    education_level: Number(form.elements.education_level.value),
    previous_turnovers: Number(form.elements.previous_turnovers.value),
    family_status: Number(form.elements.family_status.value),
    housing_type: Number(form.elements.housing_type.value),
    has_transport: Boolean(form.elements.has_transport.checked),
  };
}

function renderDetails(candidate, shiftLabel) {
  const detailsContainer = document.querySelector('#candidateDetails');

  detailsContainer.innerHTML = Object.entries(candidate)
    .map(([key, value]) => {
      const label = detailLabels[key] || key;
      let displayValue = value;

      if (key === 'shift_preference') {
        displayValue = shiftLabel || shiftLabels[value] || 'Не указано';
      }

      if (key === 'education_level') {
        displayValue = educationLabels[value] || value;
      }

      if (key === 'family_status') {
        displayValue = familyLabels[value] || value;
      }

      if (key === 'housing_type') {
        displayValue = housingLabels[value] || value;
      }

      if (key === 'has_transport') {
        displayValue = value ? 'Да' : 'Нет';
      }

      if (key === 'has_certifications') {
        displayValue = value ? 'Да' : 'Нет';
      }

      if (key === 'previous_turnovers') {
        displayValue = `${value}`;
      }

      if (key === 'years_experience') {
        displayValue = `${value} лет`;
      }

      if (key === 'age') {
        displayValue = `${value} лет`;
      }

      if (key === 'commute_time_minutes') {
        displayValue = `${value} минут`;
      }

      if (key === 'salary_expectation') {
        displayValue = `${currency(value)} ₽`;
      }

      return `
        <tr>
          <td>${label}</td>
          <td>${displayValue}</td>
        </tr>
      `;
    })
    .join('');
}

function renderRiskFactors(result) {
  const container = document.querySelector('#riskFactors');

  if (result.requires_review) {
    container.innerHTML = `
      <strong>Ключевой фактор риска</strong><br />
      Отсутствуют или неполны данные об опыте работы.
    `;
    return;
  }

  const factors = result.display_risk_factors || result.risk_factors || [];

  if (!factors.length) {
    container.textContent = 'Значительных факторов риска не обнаружено.';
    return;
  }

  container.innerHTML = `
    <ul class="risk-list">
      ${factors.map((factor) => `<li>${factor}</li>`).join('')}
    </ul>
  `;
}

function renderPositiveFactors(result) {
  const container = document.querySelector('#positiveFactors');
  const factors = result.positive_factors || [];

  if (!factors.length) {
    container.textContent = 'Значимых положительных факторов не найдено.';
    return;
  }

  container.innerHTML = `
    <ul class="risk-list positive-list">
      ${factors.map((factor) => `<li>${factor}</li>`).join('')}
    </ul>
  `;
}

function renderResult(result) {
  const theme = getTheme(result);
  const probability = clamp(result.retention_probability);
  const uncertainty = getUncertainty(result, probability);

  document.querySelector('#emptyState').classList.add('hidden');
  document.querySelector('#resultView').classList.remove('hidden');

  document.querySelector('#candidateName').textContent = state.currentPresetLabel;
  document.querySelector('#retentionScore').textContent = percent(probability);
  document.querySelector('#riskLabel').textContent = result.risk_label || theme.label;
  document.querySelector('#decisionShort').textContent = result.risk_short || theme.short;
  document.querySelector('#zoneText').textContent = result.zone_text;

  const scoreBar = document.querySelector('#scoreBar');
  scoreBar.style.width = `${Math.round(probability * 100)}%`;
  scoreBar.style.background = theme.bar;

  const scoreDial = document.querySelector('#scoreDial');
  scoreDial.style.setProperty('--score', `${Math.round(probability * 100)}%`);
  scoreDial.style.setProperty('--accent', theme.bar);

  const badge = document.querySelector('#riskBadge');
  badge.textContent = result.risk_label || theme.label;
  badge.className = `badge ${theme.badge}`;

  const confidenceBand = document.querySelector('#confidenceBand');
  confidenceBand.innerHTML = `
    <strong>Ориентировочный коридор: ${percent(uncertainty.low)} — ${percent(uncertainty.high)}</strong>
    <small>${uncertainty.note}</small>
  `;

  const decision = document.querySelector('#decisionBox');
  decision.textContent = result.status_text;
  decision.className = `decision-box ${theme.badge}`;

  renderRiskFactors(result);
  renderPositiveFactors(result);
  renderDetails(result.candidate, result.shift_label);
}

async function loadStatus() {
  const runtimeStatus = document.querySelector('#runtimeStatus');
  const dot = runtimeStatus.querySelector('.status-dot');

  try {
    const status = await apiFetch('/api/demo/status');

    dot.classList.add('ready');

    runtimeStatus.querySelector('.status-pill span:last-child').textContent =
      `Модель готова · ${status.dataset_rows} строк в датасете`;
  } catch (error) {
    runtimeStatus.querySelector('.status-pill span:last-child').textContent =
      'Статус модели недоступен';

    showToast(error.message);
  }
}

async function loadPreset(category) {
  try {
    const payload = await apiFetch(`/api/demo/candidate/${category}`);

    state.currentCandidate = payload.candidate;
    state.currentPresetLabel = presetLabels[category] || 'Демо-кандидат';

    fillCandidateForm(payload.candidate);

    await predictCurrentCandidate();
  } catch (error) {
    showToast(error.message);
  }
}

async function predictCurrentCandidate() {
  const candidate = readCandidateForm();

  state.currentCandidate = candidate;

  const result = await apiFetch('/api/demo/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(candidate),
  });

  renderResult(result);
}

function setupTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((item) => {
        item.classList.remove('is-active');
      });

      document.querySelectorAll('.tab-panel').forEach((panel) => {
        panel.classList.remove('is-active');
      });

      tab.classList.add('is-active');

      document
        .querySelector(`#tab-${tab.dataset.tab}`)
        .classList.add('is-active');

      if (tab.dataset.tab === 'history') {
        loadHistory();
      }
    });
  });
}

function setupDemo() {
  document.querySelectorAll('[data-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      loadPreset(button.dataset.preset);
    });
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    state.currentPresetLabel = 'Ручной ввод';

    try {
      await predictCurrentCandidate();
    } catch (error) {
      showToast(error.message);
    }
  });
}

function renderUploadResult(result) {
  const container = document.querySelector('#uploadResult');

  container.classList.remove('hidden');

  container.innerHTML = `
    <h3>${result.full_name}</h3>
    <p>${result.raw_summary}</p>
    <p><strong>Прогноз удержания:</strong> ${percent(result.retention_score)}</p>
    <p>
      <strong>Факторы риска:</strong>
      ${(result.risk_factors || []).join(', ') || 'не обнаружены'}
    </p>
  `;
}

function setupUpload() {
  const uploadForm = document.querySelector('#uploadForm');

  uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const fileInput = uploadForm.elements.file;

    if (!fileInput.files.length) {
      return;
    }

    const data = new FormData();
    data.append('file', fileInput.files[0]);

    const button = uploadForm.querySelector('button[type="submit"]');
    const originalText = button.textContent;

    button.disabled = true;
    button.textContent = 'Анализ выполняется...';

    try {
      const result = await apiFetch('/api/analyze', {
        method: 'POST',
        body: data,
      });

      renderUploadResult(result);
    } catch (error) {
      showToast(error.message);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
  });
}

let isChatActive = false;
let remoteAudioWs = null;
let peerConnection = null;
let localStream = null;
let lineCursor;

const remoteAudio = document.getElementById('remoteAudio');
const dialogButton = document.getElementById('startInterview');
const dialogBox = document.getElementById('dialogBox')
const remoteAudioWsUrl = `${wsBaseUrl}/webrtc`
const rtcConfig = {
    iceServers:[
      {
        urls: [
          "stun:stun.cloudflare.com:3478",
          "stun:stun.sipnet.ru:3478",
          "stun:stun.nextcloud.com:443",
        ]
      }
    ]
};

function teardownVoice() {
    console.log("Очистка старых ресурсов...");

    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }

    if (peerConnection) {
        peerConnection.onicecandidate = null;
        peerConnection.ontrack = null;
        peerConnection.close();
        peerConnection = null;
    }

    if (remoteAudioWs) {
        remoteAudioWs.onopen = null;
        remoteAudioWs.onmessage = null;
        remoteAudioWs.onerror = null;
        remoteAudioWs.onclose = null;
        
        if (remoteAudioWs.readyState !== WebSocket.CLOSED) {
            remoteAudioWs.close();
        }
        remoteAudioWs = null;
    }
}

async function setupVoice() {
    try {
        teardownVoice();
        
        dialogBox.innerHTML = ""
        remoteAudioWs = new WebSocket(remoteAudioWsUrl);

        await new Promise((resolve, reject) => {
          remoteAudioWs.onopen = () => {
            setTimeout(() => {
              if (remoteAudioWs.readyState === WebSocket.OPEN) {
                isAccepted = true;
                resolve();
              }
            }, 100);
          }
          remoteAudioWs.onerror = (err) => reject(new Error("Ошибка подключения WebSocket"));
          remoteAudioWs.onclose = (event) => {
            const reason = event.reason || "Без объяснения причины";
            if (event.code === 1013) {
              reject(new Error(`Попробуйте позже: ${reason}`));
            } else {
              reject(new Error(`WebSocket закрылся до установки соединения. Код: ${event.code}, Причина: ${reason}`));
            }
          };
        });

        console.log("WebSocket подключен");

        remoteAudioWs.onmessage = async (event) => {
          const data = JSON.parse(event.data);
          
          if (data.type === 'answer') {
            await peerConnection.setRemoteDescription(new RTCSessionDescription(data));
          } else if (data.type === 'ice-candidate') {
            await peerConnection.addIceCandidate(data.candidate);
          } else if (data.type === 'interview-user') {
            lineCursor = document.createElement('p')
            lineCursor.textContent = "Вы: — "
            dialogBox.appendChild(lineCursor)
          } else if (data.type === 'interview-model') {
            lineCursor = document.createElement('p')
            lineCursor.textContent = "Интервьюер: — "
            dialogBox.appendChild(lineCursor)
          } else if (data.type === 'interview-chunk') {
            lineCursor.textContent += data.content
          } else if (data.type === 'interview-finalize') {
            lineCursor.textContent = data.content
          } else if (data.type === 'interview-end') {
            teardownVoice()
            dialogButton.textContent = 'Начать интервью';
            dialogButton.classList.remove('recording')
            dialogButton.classList.add('primary')
          } else if (data.type === 'custom-toast') {
            showToast(data.content)
          }
        };

        localStream = await navigator.mediaDevices.getUserMedia({
          audio: true /* {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
          } */
        });
        peerConnection = new RTCPeerConnection(rtcConfig);
        
        localStream.getTracks().forEach(track => {
          peerConnection.addTrack(track, localStream);
        });

        peerConnection.onicecandidate = (event) => {
          if (event.candidate && remoteAudioWs.readyState === WebSocket.OPEN) {
            remoteAudioWs.send(JSON.stringify({ type: 'ice-candidate', candidate: event.candidate }));
          }
        };

        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        
        remoteAudioWs.send(JSON.stringify({
          type: 'offer',
          sdp: offer.sdp
        }));

        console.log("Звонок начат");

        remoteAudioWs.onclose = () => {
          showToast("Связь с сервером потеряна");
          teardownVoice();
        };

        peerConnection.ontrack = (event) => {
          console.log("Получен трек:", event.streams[0]);
          remoteAudio.srcObject = event.streams[0];
          remoteAudio.play().catch(e => console.error("Ошибка автовоспроизведения:", e));
        };

    } catch (error) {
        teardownVoice();
        throw error;
    }
}

async function voiceButtonHandler(event) {
  event.preventDefault();

  if (!isChatActive) {
    try {
      await setupVoice();
      isChatActive = true;
      dialogButton.textContent = 'Завершить интервью';
      dialogButton.classList.remove('primary');
      dialogButton.classList.add('recording');
      dialogBox.classList.remove('hidden')
  } catch (error) {
      console.error("Не удалось запустить интервью:", error);
        showToast(`Не удалось запустить интервью:\n${error}`);
      console.log(error)
    }
  }

  else {
    teardownVoice();
    isChatActive = false;
    dialogButton.textContent = 'Начать интервью';
    dialogButton.classList.remove('recording');
    dialogButton.classList.add('primary');
  }
}

function setupInterview() {
  dialogButton.addEventListener('click', voiceButtonHandler);
}

function renderHistory(items) {
  const container = document.querySelector('#historyList');

  if (!items.length) {
    container.textContent = 'История пуста.';
    container.classList.add('muted');
    return;
  }

  container.classList.remove('muted');

  container.innerHTML = items
    .map((item) => {
      const risks =
        item.risk_factors && item.risk_factors.length
          ? item.risk_factors.join(', ')
          : 'не обнаружены';

      return `
        <article class="history-item">
          <h3>${item.full_name}</h3>
          <p class="muted">${item.raw_summary}</p>
          <p><strong>Прогноз удержания:</strong> ${percent(item.retention_score)}</p>
          <p><strong>Факторы риска:</strong> ${risks}</p>
        </article>
      `;
    })
    .join('');
}

async function loadHistory() {
  const container = document.querySelector('#historyList');

  container.textContent = 'Загрузка истории...';
  container.classList.add('muted');

  try {
    const items = await apiFetch('/api/history');
    renderHistory(items);
  } catch (error) {
    container.textContent = 'Не удалось загрузить историю.';
    showToast(error.message);
  }
}

function setupHistory() {
  document.querySelector('#refreshHistory').addEventListener('click', loadHistory);
}

function setInitialCandidate() {
  const candidate = {
    skills_verified_count: 6,
    years_experience: 3.5,
    age: 31,
    commute_time_minutes: 45,
    shift_preference: 2,
    salary_expectation: 90000,
    has_certifications: true,
    education_level: 1,
    previous_turnovers: 1,
    family_status: 0,
    housing_type: 1,
    has_transport: true,
  };

  state.currentCandidate = candidate;
  fillCandidateForm(candidate);
}

window.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupDemo();
  setupUpload();
  setupHistory();
  setupInterview();
  setInitialCandidate();
  loadStatus();
});