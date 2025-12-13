<template>
  <div class="min-h-screen bg-brand-dark font-sans text-gray-100">
    <NavBar v-if="isAuthenticated && user" :user="user" @force-sync="forceSync" />
    
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, provide } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import axios from 'axios'
import NavBar from './components/NavBar.vue'
import i18n from './i18n'

const user = ref(null)
const isAuthenticated = ref(false)
const absUrl = ref('')
const router = useRouter()
const route = useRoute()

provide('absUrl', absUrl)

const checkAuth = async () => {
  try {
    const { data } = await axios.get('/api/auth/status')
    isAuthenticated.value = data.authenticated
    user.value = data.user
    
    if (data.abs_url) {
      absUrl.value = data.abs_url
    }
    
    if (user.value && user.value.language) {
      const { locale } = i18n.global
      locale.value = user.value.language
    }
    
    // If on login page and authenticated, redirect home or admin
    if (route.path === '/login' && isAuthenticated.value) {
      if (user.value.id === 'root') {
        router.push('/admin')
      } else {
        router.push('/')
      }
    }
  } catch (error) {
    isAuthenticated.value = false
    user.value = null
    // If not on login page and not authenticated, redirect login (unless public? No, all is protected)
    if (route.path !== '/login') {
      router.push('/login')
    }
  }
}

provide('checkAuth', checkAuth)

const forceSync = async () => {
  if (!confirm('Are you sure you want to force sync the library? This might take a while.')) return;
  
  try {
    await axios.post('/api/admin/force-sync')
    alert('Sync triggered successfully!')
  } catch (err) {
    alert('Error triggering sync: ' + (err.response?.data?.error || err.message))
  }
}

onMounted(() => {
  checkAuth()
})

// Enforce strict routing rules and auth checks
watch([() => user.value, () => route.path], ([newUser, newPath]) => {
  if (newUser) {
      if (newUser.force_password_change && newPath !== '/settings') {
          router.replace('/settings')
      } else if (newUser.id === 'root') {
          const allowedPaths = ['/admin', '/settings', '/login'] // login is allowed as we might be redirecting out of it
          if (!allowedPaths.includes(newPath)) {
              router.replace('/admin')
          }
      }
  }
})
</script>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
