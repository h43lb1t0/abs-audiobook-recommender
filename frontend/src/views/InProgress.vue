<template>
  <div class="space-y-8 animate-fade-in">
    <div class="bg-brand-card p-6 rounded-md shadow-sm">
      <h2 class="text-2xl font-bold text-white mb-2">Currently Reading</h2>
      <p class="text-gray-400 font-medium" v-if="loading">Loading books...</p>
      <p class="text-gray-400 font-medium" v-else>{{ books.length }} book{{ books.length !== 1 ? 's' : '' }} in progress</p>
    </div>

    <div v-if="loading" class="flex justify-center py-20">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-brand-primary"></div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-center">
      {{ error }}
    </div>

    <template v-else>
       <!-- Series Groups -->
       <div v-for="(group, seriesName) in seriesGroups" :key="seriesName" class="space-y-4">
         <div class="flex items-center gap-4 py-2 border-b border-white/5 px-4 -mx-4">
           <h3 class="text-xl font-bold text-white">{{ seriesName }}</h3>
           <span class="text-xs font-mono bg-slate-800 text-gray-400 px-2 py-1 rounded-full">{{ group.length }} books</span>
         </div>
         <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
           <BookCard v-for="book in group" :key="book.id" :book="book">
             <template #actions>
               <button v-if="book.status === 'abandoned'" @click="updateStatus(book, 'reading')"
                 class="w-full py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm font-bold tracking-wide transition-all shadow hover:shadow-lg">
                 Reactivate
               </button>
               <button v-else @click="updateStatus(book, 'abandoned')"
                 class="w-full py-2 bg-slate-800/50 hover:bg-red-900/30 text-slate-400 hover:text-red-300 border border-slate-700 hover:border-red-900/50 rounded-lg text-sm font-semibold transition-all">
                 Abandon
               </button>
             </template>
           </BookCard>
         </div>
       </div>

       <!-- Standalone -->
       <div v-if="standaloneBooks.length > 0" class="space-y-4">
         <div class="flex items-center gap-4 py-2 border-b border-white/5 bg-gradient-to-r from-white/5 to-transparent px-4 -mx-4">
           <h3 class="text-xl font-bold text-white">Standalone Books</h3>
           <span class="text-xs font-mono bg-slate-800 text-gray-400 px-2 py-1 rounded-full">{{ standaloneBooks.length }} books</span>
         </div>
         <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
           <BookCard v-for="book in standaloneBooks" :key="book.id" :book="book">
             <template #actions>
               <button v-if="book.status === 'abandoned'" @click="updateStatus(book, 'reading')"
                 class="w-full py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm font-bold tracking-wide transition-all shadow hover:shadow-lg">
                 Reactivate
               </button>
               <button v-else @click="updateStatus(book, 'abandoned')"
                 class="w-full py-2 bg-slate-800/50 hover:bg-red-900/30 text-slate-400 hover:text-red-300 border border-slate-700 hover:border-red-900/50 rounded-lg text-sm font-semibold transition-all">
                 Abandon
               </button>
             </template>
           </BookCard>
         </div>
       </div>
       
       <div v-if="books.length === 0" class="text-center py-24">
         <p class="text-gray-400 text-lg">No books in progress.</p>
       </div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import BookCard from '@/components/BookCard.vue'

const books = ref([])
const loading = ref(true)
const error = ref('')

const seriesGroups = computed(() => {
  const groups = {}
  books.value.forEach(book => {
    if (book.series) {
      if (!groups[book.series]) groups[book.series] = []
      groups[book.series].push(book)
    }
  })
  // Sort keys alphabetically ignoring case
  return Object.keys(groups).sort((a,b) => a.localeCompare(b, undefined, {sensitivity: 'base'})).reduce((acc, key) => {
    acc[key] = groups[key]
    return acc
  }, {})
})

const standaloneBooks = computed(() => {
  return books.value.filter(b => !b.series)
})

const fetchBooks = async () => {
  try {
    const { data } = await axios.get('/api/in-progress')
    books.value = data
  } catch (err) {
    error.value = err.response?.data?.error || err.message
  } finally {
    loading.value = false
  }
}

const updateStatus = async (book, newStatus) => {
  const endpoint = newStatus === 'abandoned' ? '/api/abandon-book' : '/api/reactivate-book'
  const actionName = newStatus === 'abandoned' ? 'abandon' : 'reactivate'
  
  if (!confirm(`Are you sure you want to ${actionName} this book?`)) return;

  // Optimistic update
  const originalStatus = book.status
  book.status = newStatus
  
  try {
    await axios.post(endpoint, { book_id: book.id })
  } catch (err) {
    // Revert
    book.status = originalStatus
    alert(`Failed to ${actionName} book: ` + (err.response?.data?.error || err.message))
  }
}

onMounted(fetchBooks)
</script>
