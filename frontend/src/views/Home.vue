<template>
  <div class="space-y-8 animate-fade-in">
    <!-- Header / Controls -->
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-800/30 p-6 rounded-2xl border border-white/5 backdrop-blur-sm shadow-xl">
      <div>
        <h2 class="text-2xl font-bold text-white mb-2 tracking-tight">My Recommendations</h2>
        <p class="text-gray-400 font-medium">Discover your next favorite adventure</p>
      </div>
      
      <div class="flex flex-col items-end gap-2">
        <button @click="fetchRecommendations(true)" :disabled="loading" 
          class="flex items-center gap-2 bg-gradient-to-r from-brand-primary to-brand-accent hover:from-brand-primary/90 hover:to-brand-accent/90 text-white px-6 py-3 rounded-xl font-bold transition-all duration-300 shadow-lg hover:shadow-brand-primary/25 disabled:opacity-70 disabled:cursor-not-allowed hover:-translate-y-0.5">
          <svg v-if="loading" class="animate-spin h-5 w-5" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span v-else class="flex items-center gap-2"><svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg> Generate New</span>
        </button>
        <span v-if="lastGenerated" class="text-xs text-gray-500 font-mono bg-black/20 px-2 py-1 rounded">
          Last updated: {{ formatDate(lastGenerated) }}
        </span>
      </div>
    </div>

    <!-- Error State -->
    <div v-if="error" class="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-center flex items-center justify-center gap-2">
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" /></svg>
      {{ error }}
    </div>

    <!-- Grid -->
    <div v-if="recommendations.length > 0" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
      <BookCard v-for="book in recommendations" :key="book.id" :book="book">
        <template #content>
          <div class="mt-3 text-sm text-gray-300 italic relative pl-4 leading-relaxed">
            <span class="absolute left-0 top-0 text-brand-primary text-lg opacity-50">"</span>
            {{ book.reason }}
            <span class="absolute bottom-0 text-brand-primary text-lg opacity-50">"</span>
          </div>
        </template>
      </BookCard>
    </div>

    <!-- Empty State -->
    <div v-else-if="!loading && !error" class="text-center py-24 bg-slate-800/20 rounded-3xl border border-dashed border-white/5">
      <div class="inline-flex items-center justify-center p-6 rounded-full bg-slate-800/50 mb-4 ring-1 ring-white/10">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      </div>
      <p class="text-gray-400 text-lg font-medium">Tap "Generate New" to get started</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { io } from 'socket.io-client'
import BookCard from '@/components/BookCard.vue'

const socket = ref(null)
const recommendations = ref([])
const loading = ref(false)
const error = ref('')
const lastGenerated = ref(null)

const formatDate = (dateStr) => {
  return new Date(dateStr).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: 'numeric', minute: '2-digit'
  })
}

const fetchRecommendations = (refresh = false) => {
  loading.value = true
  error.value = ''
  // Don't clear recommendations immediately on refresh to avoid flash, unless strict requirement.
  // Actually, standard UX: show loading state on button, keep content until new arrives.
  // But if it's a completely new set, maybe clearing is honest.
  // I'll keep them but show loading.
  
  socket.value.emit('get_recommendations', { refresh })
}

onMounted(() => {
  // Use relative path for socket.io to work with proxy and production
  socket.value = io({
    path: '/socket.io',
    transports: ['websocket', 'polling']
  })

  socket.value.on('connect', () => {
    // Initial fetch (cache only)
    fetchRecommendations(false)
  })

  socket.value.on('recommendations_ready', (data) => {
    loading.value = false
    recommendations.value = data.recommendations || []
    lastGenerated.value = data.generated_at
  })

  socket.value.on('error', (data) => {
    loading.value = false
    error.value = data.error || 'Failed to fetch recommendations'
  })
})

onUnmounted(() => {
  if (socket.value) socket.value.disconnect()
})
</script>
