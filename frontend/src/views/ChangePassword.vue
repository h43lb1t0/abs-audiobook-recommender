<template>
  <div class="max-w-md mx-auto space-y-8 animate-fade-in bg-brand-card p-8 rounded-md shadow-sm mt-8">
    <div>
      <h2 class="text-2xl font-bold text-white">{{ $t('changePassword.title') }}</h2>
      <p class="text-gray-400 text-sm mt-1">{{ $t('changePassword.subtitle') }}</p>
      
      <div v-if="isForced" class="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg text-yellow-200 text-sm flex items-center">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
        </svg>
        <span>{{ $t('changePassword.forcedMessage') }}</span>
      </div>
    </div>

    <form @submit.prevent="updatePassword" class="space-y-6">
      <div v-if="message" :class="`p-4 rounded-lg text-sm ${isError ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-green-500/10 text-green-400 border border-green-500/20'}`">
        {{ message }}
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-300 mb-1">{{ $t('changePassword.currentPassword') }}</label>
        <input type="password" v-model="currentPassword" required class="block w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-brand-primary focus:border-transparent transition-all">
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-300 mb-1">{{ $t('changePassword.newPassword') }}</label>
        <input type="password" v-model="newPassword" required class="block w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-brand-primary focus:border-transparent transition-all">
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-300 mb-1">{{ $t('changePassword.confirmNewPassword') }}</label>
        <input type="password" v-model="confirmPassword" required class="block w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-brand-primary focus:border-transparent transition-all">
      </div>
      
      <button type="submit" :disabled="loading" class="w-full bg-brand-primary hover:bg-brand-primary/90 text-white font-bold py-3 px-4 rounded-lg transition-all shadow-lg hover:shadow-brand-primary/25 disabled:opacity-50">
        {{ loading ? $t('changePassword.updating') : $t('changePassword.updateButton') }}
      </button>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'

const currentPassword = ref('')
const newPassword = ref('')
const confirmPassword = ref('')
const loading = ref(false)
const message = ref('')
const isError = ref(false)
const router = useRouter()
const { t } = useI18n()

import { inject } from 'vue'

const isForced = ref(false)
const checkAuth = inject('checkAuth')

const checkForced = async () => {
    try {
        const { data } = await axios.get('/api/auth/status')
        if (data.authenticated && data.user.force_password_change) {
            isForced.value = true
        }
    } catch (e) {
        // ignore
    }
}
checkForced()

const updatePassword = async () => {
  if (newPassword.value.length < 4) {
    message.value = t('changePassword.tooShort')
    isError.value = true
    return
  }

  if (newPassword.value !== confirmPassword.value) {
    message.value = t('changePassword.mismatch')
    isError.value = true
    return
  }

  loading.value = true
  message.value = ''
  isError.value = false

  try {
    const response = await axios.post('/change-password', {
      current_password: currentPassword.value,
      new_password: newPassword.value,
      confirm_password: confirmPassword.value
    }) 
    
    if (response.data.success || response.status === 200) { 
        message.value = t('changePassword.success')
        isError.value = false
        // Clear fields
        currentPassword.value = ''
        newPassword.value = ''
        confirmPassword.value = ''
        
        // Refresh auth state to update force_password_change in App.vue
        if (checkAuth) {
            await checkAuth()
        }
        
        setTimeout(() => router.push('/'), 1500)
    }
  } catch (err) {
    isError.value = true
    message.value = err.response?.data?.error || t('changePassword.error')
  } finally {
    loading.value = false
  }
}
</script>
