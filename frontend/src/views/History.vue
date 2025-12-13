<template>
  <div class="space-y-8 animate-fade-in">
    <div class="bg-slate-800/30 p-6 rounded-2xl border border-white/5 backdrop-blur-sm shadow-xl">
      <h2 class="text-2xl font-bold text-white mb-2">Listening History</h2>
      <p class="text-gray-400 font-medium" v-if="loading">Loading history...</p>
      <p class="text-gray-400 font-medium" v-else>{{ books.length }} book{{ books.length !== 1 ? 's' : '' }} finished &middot; {{ ratedCount }} rated</p>
    </div>

    <div v-if="loading" class="flex justify-center py-20">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-brand-primary"></div>
    </div>

    <div v-else-if="error" class="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-center">
      {{ error }}
    </div>

    <template v-else>
       <!-- Series Groups -->
       <div v-for="(group, seriesName) in seriesGroups" :key="seriesName" class="space-y-4">
         <div class="flex items-center gap-4 py-2 border-b border-white/5 bg-gradient-to-r from-white/5 to-transparent px-4 -mx-4">
           <h3 class="text-xl font-bold text-white">{{ seriesName }}</h3>
           <span class="text-xs font-mono bg-slate-800 text-gray-400 px-2 py-1 rounded-full">{{ group.length }} books</span>
         </div>
         <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
           <BookCard v-for="book in group" :key="book.id" :book="book">
             <template #content>
               <div class="mt-3 flex justify-center py-2 bg-slate-900/40 rounded-lg">
                 <StarRating :rating="book.rating || 0" @update:rating="val => rateBook(book, val)" />
               </div>
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
             <template #content>
               <div class="mt-3 flex justify-center py-2 bg-slate-900/40 rounded-lg">
                 <StarRating :rating="book.rating || 0" @update:rating="val => rateBook(book, val)" />
               </div>
             </template>
           </BookCard>
         </div>
       </div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import BookCard from '@/components/BookCard.vue'
import StarRating from '@/components/StarRating.vue'

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
  return Object.keys(groups).sort((a,b) => a.localeCompare(b, undefined, {sensitivity: 'base'})).reduce((acc, key) => {
    acc[key] = groups[key]
    return acc
  }, {})
})

const standaloneBooks = computed(() => {
  return books.value.filter(b => !b.series)
})

const ratedCount = computed(() => {
  return books.value.filter(b => b.rating > 0).length
})

const fetchHistory = async () => {
  try {
    const { data } = await axios.get('/api/listening-history')
    books.value = data
  } catch (err) {
    error.value = err.response?.data?.error || err.message
  } finally {
    loading.value = false
  }
}

const rateBook = async (book, rating) => {
  const originalRating = book.rating
  book.rating = rating
  
  try {
    await axios.post('/api/rate-book', { book_id: book.id, rating })
  } catch (err) {
    book.rating = originalRating
    alert('Failed to save rating: ' + (err.response?.data?.error || err.message))
  }
}

onMounted(fetchHistory)
</script>
