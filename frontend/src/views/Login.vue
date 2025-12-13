<template>
  <div class="min-h-screen flex items-center justify-center bg-brand-dark px-4 py-12 sm:px-6 lg:px-8 relative overflow-hidden">
    <!-- Background Decor -->
    <div class="absolute top-0 left-0 w-full h-full overflow-hidden z-0">
      <div class="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-brand-primary/10 blur-[100px]"></div>
      <div class="absolute top-[30%] right-[10%] w-[30%] h-[30%] rounded-full bg-brand-accent/10 blur-[80px]"></div>
    </div>

    <div class="max-w-md w-full space-y-8 bg-slate-900/80 backdrop-blur-xl p-8 rounded-2xl shadow-2xl border border-white/10 relative z-10">
      <div class="text-center">
        <h2 class="mt-2 text-3xl font-bold text-white tracking-tight">
          Welcome Back
        </h2>
        <p class="mt-2 text-sm text-gray-400">
          Sign in to access your audiobook recommendations
        </p>
      </div>
      <form class="mt-8 space-y-6" @submit.prevent="handleLogin">
        <div class="space-y-4">
          <div>
            <label for="username" class="block text-sm font-medium text-gray-300 mb-1">Username</label>
            <input id="username" name="username" type="text" v-model="username" required 
              class="appearance-none block w-full px-4 py-3 border border-gray-700/50 placeholder-gray-500 text-white rounded-xl bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-brand-primary focus:border-transparent focus:bg-gray-800 transition-all duration-200" 
              placeholder="Enter your username">
          </div>
          <div>
            <label for="password" class="block text-sm font-medium text-gray-300 mb-1">Password</label>
            <input id="password" name="password" type="password" v-model="password" required 
              class="appearance-none block w-full px-4 py-3 border border-gray-700/50 placeholder-gray-500 text-white rounded-xl bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-brand-primary focus:border-transparent focus:bg-gray-800 transition-all duration-200" 
              placeholder="Enter your password">
          </div>
        </div>

        <div v-if="error" class="text-red-400 text-sm text-center bg-red-500/10 p-3 rounded-xl border border-red-500/20 animate-pulse">
          {{ error }}
        </div>

        <div>
          <button type="submit" :disabled="loading"
            class="group relative w-full flex justify-center py-3.5 px-4 border border-transparent text-sm font-bold rounded-xl text-white bg-brand-primary hover:bg-brand-primary/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-primary disabled:opacity-70 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-brand-primary/30 hover:-translate-y-0.5">
            <span v-if="loading" class="absolute left-0 inset-y-0 flex items-center pl-3">
              <svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            </span>
            {{ loading ? 'Signing in...' : 'Sign in' }}
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'

const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')
const router = useRouter()

const handleLogin = async () => {
  loading.value = true
  error.value = ''
  
  try {
    const response = await axios.post('/login', {
      username: username.value,
      password: password.value
    })
    
    if (response.data.success) {
      // Reload to ensure auth state is propagated everywhere cleanly
      window.location.href = '/'
    }
  } catch (err) {
    if (err.response && err.response.data && err.response.data.error) {
      error.value = err.response.data.error
    } else {
      error.value = 'An unexpected error occurred'
    }
  } finally {
    loading.value = false
  }
}
</script>
