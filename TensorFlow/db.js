// Configure Firebase with the web app settings from Firebase Console.
const firebaseConfig = {
    apiKey: 'AIzaSyD0oNKPoa2S5YMOxk2xzzxSa_Xc5pbZZQs',
    authDomain: 'p2crud007.firebaseapp.com',
    projectId: 'p2crud007',
    storageBucket: 'p2crud007.firebasestorage.app',
    messagingSenderId: '887406012041',
    appId: '1:887406012041:web:3ae775cad29764968dc858'
};

class MockFirestore {
    constructor() {
        try {
            this.data = JSON.parse(localStorage.getItem('mock_inventory_db') || '[]');
            if (!Array.isArray(this.data)) this.data = [];
        } catch (error) {
            console.warn('Local database rusak, memulai penyimpanan baru:', error);
            this.data = [];
        }
        this.listeners = [];
    }

    async add(payload) {
        const record = { ...payload, id: String(Date.now()) };
        this.data.unshift(record);
        this.data = this.data.slice(0, 50);
        localStorage.setItem('mock_inventory_db', JSON.stringify(this.data));
        this.listeners.forEach(listener => listener([...this.data]));
        return record;
    }

    onSnapshot(listener) {
        this.listeners.push(listener);
        listener([...this.data]);
        return () => {
            this.listeners = this.listeners.filter(current => current !== listener);
        };
    }
}

const dbStatusElement = document.getElementById('db-status');
let mockDatabase = null;
let firestoreCollection = null;
let isMockMode = true;

function setDatabaseStatus(label, mode) {
    if (!dbStatusElement) return;
    dbStatusElement.textContent = label;
    dbStatusElement.className = `status-badge ${mode}`;
}

function hasConfiguredFirebase() {
    return Boolean(
        firebaseConfig.apiKey &&
        firebaseConfig.apiKey !== 'PASTE_API_KEY_ANDA_DI_SINI' &&
        firebaseConfig.authDomain !== 'your-project.firebaseapp.com' &&
        firebaseConfig.projectId &&
        firebaseConfig.projectId !== 'your-project-id' &&
        firebaseConfig.appId !== '1:123456789:web:abcdef'
    );
}

if (hasConfiguredFirebase() && typeof firebase !== 'undefined') {
    try {
        if (!firebase.apps.length) firebase.initializeApp(firebaseConfig);
        firestoreCollection = firebase.firestore().collection('detections');
        isMockMode = false;
        setDatabaseStatus('FIREBASE ONLINE', 'online');
    } catch (error) {
        console.error('Firebase init error:', error);
        setDatabaseStatus('LOCAL MODE (Firebase gagal)', 'demo');
    }
}

if (isMockMode) {
    mockDatabase = new MockFirestore();
    setDatabaseStatus('DEMO MODE (LocalStorage)', 'demo');
}

window.databaseService = {
    async addDetection(detections) {
        const now = new Date();
        const payload = {
            items: detections.map(item => `${item.class || item.name || 'objek'} (${Number(item.confidence || 0).toFixed(0)}%)`).join(', '),
            detections,
            pattern_summary: detections.map(item => `${item.class || item.name || 'objek'} (${item.color || 'unknown'}, ${item.size || 'unknown'}, ${item.shape || 'unknown'})`).join(' | '),
            count: detections.length,
            device_id: 'BROWSER-TEST',
            timestamp: isMockMode ? now.toISOString() : firebase.firestore.FieldValue.serverTimestamp()
        };

        if (isMockMode) {
            await mockDatabase.add(payload);
        } else {
            await firestoreCollection.add(payload);
        }
        return payload;
    },

    subscribe(onData, onError) {
        if (isMockMode) return mockDatabase.onSnapshot(onData);

        return firestoreCollection
            .orderBy('timestamp', 'desc')
            .limit(30)
            .onSnapshot(snapshot => {
                const records = snapshot.docs.map(document => {
                    const record = document.data();
                    const timestamp = record.timestamp;
                    return {
                        ...record,
                        id: document.id,
                        timestamp: timestamp && typeof timestamp.toDate === 'function'
                            ? timestamp.toDate().toISOString()
                            : timestamp
                    };
                });
                onData(records);
            }, error => {
                console.error('Gagal membaca data database:', error);
                if (onError) onError(error);
            });
    }
};