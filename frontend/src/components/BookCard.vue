<template>
  <div class="group relative flex flex-col bg-brand-card rounded-md overflow-hidden shadow-sm transition-all duration-200 hover:shadow-md">
    <div class="relative aspect-square w-full overflow-hidden bg-brand-dark">
      <img :src="coverUrl" :alt="book.title" loading="lazy" class="h-full w-full object-cover transition-opacity duration-300 group-hover:opacity-90" @error="handleImageError">
      
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
      <p class="text-slate-400 text-sm font-medium mb-3 line-clamp-1">{{ book.author }}</p>
      
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
import { computed } from 'vue'

const props = defineProps({
  book: {
    type: Object,
    required: true
  }
})

const coverUrl = computed(() => {
  return props.book.id ? `/api/cover/${props.book.id}` : ''
})

const handleImageError = (e) => {
  e.target.src = 'https://via.placeholder.com/300x450/0f172a/cbd5e1?text=No+Cover'
}
</script>
