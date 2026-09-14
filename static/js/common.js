// VisionCam — outils partagés par les pages : DOM, API, notifications, temps réel.

/**
 * Crée un élément. Les enfants de type chaîne deviennent des nœuds texte : un
 * nom saisi par un utilisateur n'est jamais interprété comme du HTML.
 */
export function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;
        if (key.startsWith('on') && typeof value === 'function') {
            el.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (key === 'dataset') {
            Object.assign(el.dataset, value);
        } else if (value === true) {
            el.setAttribute(key, '');
        } else {
            el.setAttribute(key, value);
        }
    }
    for (const child of children.flat()) {
        if (child === null || child === undefined || child === false) continue;
        el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
}

/** Appel à l'API : JSON en retour ; une session expirée renvoie vers la connexion. */
export async function api(path, options = {}) {
    const init = { ...options };
    if (init.json !== undefined) {
        init.body = JSON.stringify(init.json);
        init.headers = { 'Content-Type': 'application/json', ...(init.headers || {}) };
        delete init.json;
    }
    const res = await fetch(path, init);
    if (res.status === 401) {
        window.location.href = `/login?next=${encodeURIComponent(location.pathname + location.search)}`;
        throw new Error('Connexion requise.');
    }
    const data = res.headers.get('Content-Type')?.includes('application/json') ? await res.json() : {};
    return { ok: res.ok, status: res.status, data };
}

export function toast(message, kind = '') {
    const box = document.getElementById('toasts');
    const el = h('div', { class: `toast ${kind}`, role: kind === 'error' ? 'alert' : 'status' }, message);
    box.append(el);
    setTimeout(() => el.remove(), kind === 'error' ? 7000 : 4000);
}

export function formatTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function formatDateTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('fr-FR', {
        day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit',
    });
}

export function formatDuration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const hours = Math.floor(s / 3600);
    const minutes = Math.floor((s % 3600) / 60);
    if (hours) return `${hours} h ${String(minutes).padStart(2, '0')}`;
    if (minutes) return `${minutes} min ${String(s % 60).padStart(2, '0')}`;
    return `${s} s`;
}

// ── Temps réel : un seul flux Server-Sent Events par page ─────────────────

const handlers = new Map();
let source = null;

/** Abonne `handler` aux événements de type `type` ('*' pour tous). */
export function onEvent(type, handler) {
    if (!handlers.has(type)) handlers.set(type, new Set());
    handlers.get(type).add(handler);
    connect();
    return () => handlers.get(type).delete(handler);
}

function connect() {
    if (source) return;
    const pill = document.getElementById('livePill');
    source = new EventSource('/api/events');
    source.onopen = () => {
        pill?.classList.add('connected');
        if (pill) pill.textContent = 'En direct';
    };
    source.onerror = () => {
        // Le navigateur se reconnecte seul (retry envoyé par le serveur).
        pill?.classList.remove('connected');
        if (pill) pill.textContent = 'Reconnexion…';
    };
    source.onmessage = (message) => {
        let event;
        try {
            event = JSON.parse(message.data);
        } catch {
            return;
        }
        for (const key of [event.type, '*']) {
            for (const handler of handlers.get(key) || []) handler(event);
        }
    };
}

// ── Alertes du navigateur (arrivées, personnes inconnues) ─────────────────

const NOTIFY_KEY = 'visioncam.notifications';

function notificationsWanted() {
    try {
        return localStorage.getItem(NOTIFY_KEY) === 'on';
    } catch {
        return false;
    }
}

function setNotificationsWanted(on) {
    try {
        localStorage.setItem(NOTIFY_KEY, on ? 'on' : 'off');
    } catch {
        // Stockage indisponible (navigation privée) : la préférence reste pour la page.
    }
}

function notify(title, body) {
    if (!notificationsWanted() || !('Notification' in window) || Notification.permission !== 'granted') return;
    // Pas de notification quand la page est au premier plan : le fil suffit.
    if (document.visibilityState === 'visible') return;
    new Notification(title, { body, tag: `${title}-${body}` });
}

function setupNotifications() {
    const button = document.getElementById('notifyBtn');
    if (!button) return;
    if (!('Notification' in window)) {
        button.hidden = true;
        return;
    }
    const render = () => {
        const on = notificationsWanted() && Notification.permission === 'granted';
        button.textContent = on ? 'Alertes : activées' : 'Alertes : désactivées';
        button.setAttribute('aria-pressed', String(on));
    };
    button.addEventListener('click', async () => {
        if (notificationsWanted() && Notification.permission === 'granted') {
            setNotificationsWanted(false);
        } else {
            const permission = await Notification.requestPermission();
            setNotificationsWanted(permission === 'granted');
            if (permission !== 'granted') toast('Notifications refusées par le navigateur.', 'error');
        }
        render();
    });
    render();
    onEvent('arrival', (e) => notify('Arrivée', e.name));
    onEvent('unknown', () => notify('Personne inconnue', 'Une personne non reconnue est devant la caméra.'));
}

setupNotifications();
connect();
