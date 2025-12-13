<template>
  <div class="group relative flex flex-col bg-brand-card rounded-md overflow-hidden shadow-sm transition-all duration-200 hover:shadow-md">
    <div class="relative aspect-square w-full overflow-hidden bg-brand-dark">
      <a :href="itemUrl" target="_blank" rel="noopener noreferrer" class="block h-full w-full cursor-pointer" v-if="itemUrl">
        <img :src="coverUrl" :alt="book.title" loading="lazy" class="h-full w-full object-cover transition-opacity duration-300 group-hover:opacity-90" @error="handleImageError">
      </a>
      <img v-else :src="coverUrl" :alt="book.title" loading="lazy" class="h-full w-full object-cover transition-opacity duration-300 group-hover:opacity-90" @error="handleImageError">
      
      <!-- Progress Bar Overlay for In Progress -->
      <div v-if="book.progress !== undefined" class="absolute bottom-0 left-0 right-0 h-1.5 bg-black/50">
        <div class="h-full bg-brand-primary" :style="{ width: (book.progress * 100) + '%' }"></div>
      </div>
      
      <!-- Abandoned Overlay -->
      <div v-if="book.status === 'abandoned'" class="absolute inset-0 bg-black/60 flex items-center justify-center">
        <span class="px-2 py-1 bg-red-600 text-white text-xs font-bold rounded uppercase tracking-wider">Abandoned</span>
      </div>
    </div>
    
    <div class="flex flex-col flex-1 p-4">
      <h3 class="text-white font-bold text-lg leading-tight line-clamp-2 mb-1" :title="book.title">
        <span v-if="book.series_sequence" class="inline-block px-1.5 py-0.5 bg-brand-primary/20 text-brand-primary text-[10px] font-bold rounded mr-1 align-middle">
          #{{ book.series_sequence }}
        </span>
        {{ book.title }}
      </h3>
      <p class="text-slate-400 text-sm font-medium mb-1 line-clamp-1">{{ book.author }}</p>

      <!-- Duration -->
      <p v-if="book.duration" class="text-xs text-brand-primary mb-2 font-mono flex items-center gap-1">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd" />
        </svg>
        {{ book.duration }}
      </p>

      <!-- Description -->
      <p 
        v-if="book.description" 
        @click.stop.prevent="showFullDescription = !showFullDescription"
        class="text-xs mb-3 leading-relaxed cursor-pointer hover:text-slate-300 transition-colors"
        :class="{ 
          'line-clamp-3 text-slate-500': !showFullDescription,
          'text-white': showFullDescription
        }"
        :title="showFullDescription ? 'Click to show less' : 'Click to show more'"
      >
        {{ book.description }}
      </p>
      
      <!-- Slot for extra content like Reason or Rating -->
      <div class="mt-auto">
        <slot name="content"></slot>
      </div>
      
      <!-- Slot for actions (buttons) -->
      <div v-if="$slots.actions" class="mt-3 pt-3 border-t border-white/5 grid gap-2">
        <slot name="actions"></slot>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, inject, ref } from 'vue'

const props = defineProps({
  book: {
    type: Object,
    required: true
  }
})

const showFullDescription = ref(false)

const absUrl = inject('absUrl', { value: '' }) // Inject ref or default object with value

const itemUrl = computed(() => {
  return absUrl.value ? `${absUrl.value}/audiobookshelf/item/${props.book.id}` : ''
})

const coverUrl = computed(() => {
  return props.book.id ? `/api/cover/${props.book.id}` : ''
})

const handleImageError = (e) => {
  e.target.src = 'https://via.placeholder.com/300x450/0f172a/cbd5e1?text=No+Cover'
}
</script>
