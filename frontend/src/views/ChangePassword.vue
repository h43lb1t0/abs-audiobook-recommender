<template>
  <div class="max-w-md mx-auto space-y-8 animate-fade-in bg-brand-card p-8 rounded-md shadow-sm mt-8">
    <div>
      <h2 class="text-2xl font-bold text-white">Change Password</h2>
      <p class="text-gray-400 text-sm mt-1">Update your account credentials</p>
    </div>

    <form @submit.prevent="updatePassword" class="space-y-6">
      <div v-if="message" :class="`p-4 rounded-lg text-sm ${isError ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-green-500/10 text-green-400 border border-green-500/20'}`">
        {{ message }}
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-300 mb-1">Current Password</label>
        <input type="password" v-model="currentPassword" required class="block w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-brand-primary focus:border-transparent transition-all">
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-300 mb-1">New Password</label>
        <input type="password" v-model="newPassword" required class="block w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-brand-primary focus:border-transparent transition-all">
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-300 mb-1">Confirm New Password</label>
        <input type="password" v-model="confirmPassword" required class="block w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-brand-primary focus:border-transparent transition-all">
      </div>
      
      <button type="submit" :disabled="loading" class="w-full bg-brand-primary hover:bg-brand-primary/90 text-white font-bold py-3 px-4 rounded-lg transition-all shadow-lg hover:shadow-brand-primary/25 disabled:opacity-50">
        {{ loading ? 'Updating...' : 'Update Password' }}
      </button>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'

const currentPassword = ref('')
const newPassword = ref('')
const confirmPassword = ref('')
const loading = ref(false)
const message = ref('')
const isError = ref(false)
const router = useRouter()

const updatePassword = async () => {
  if (newPassword.value !== confirmPassword.value) {
    message.value = 'New passwords do not match'
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
        message.value = 'Password updated successfully'
        isError.value = false
        // Clear fields
        currentPassword.value = ''
        newPassword.value = ''
        confirmPassword.value = ''
        
        setTimeout(() => router.push('/'), 1500)
    }
  } catch (err) {
    isError.value = true
    message.value = err.response?.data?.error || 'Failed to update password'
  } finally {
    loading.value = false
  }
}
</script>
