import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router'

import i18n from './i18n'

import axios from 'axios'
import { watch } from 'vue'

// Set initial header
axios.defaults.headers.common['Accept-Language'] = i18n.global.locale.value

// Watch for language changes
watch(
    () => i18n.global.locale.value,
    (newLocale) => {
        axios.defaults.headers.common['Accept-Language'] = newLocale
    }
)

createApp(App).use(router).use(i18n).mount('#app')
