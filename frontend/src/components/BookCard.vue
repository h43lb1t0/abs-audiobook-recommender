<template>
  <div class="group relative flex flex-col bg-slate-800/50 backdrop-blur-md rounded-xl overflow-hidden border border-white/5 shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-brand-primary/20">
    <div class="relative aspect-[2/3] w-full overflow-hidden bg-slate-900">
      <img :src="coverUrl" :alt="book.title" loading="lazy" class="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105" @error="handleImageError">
      
      <!-- Progress Bar Overlay for In Progress -->
      <div v-if="book.progress !== undefined" class="absolute bottom-0 left-0 right-0 h-1.5 bg-slate-700/50 backdrop-blur-sm">
        <div class="h-full bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)] transition-all duration-500" :style="{ width: (book.progress * 100) + '%' }"></div>
      </div>
      
      <!-- Abandoned Overlay -->
      <div v-if="book.status === 'abandoned'" class="absolute inset-0 bg-black/60 flex items-center justify-center backdrop-blur-[1px]">
        <span class="px-3 py-1 bg-red-500/80 text-white text-xs font-bold rounded-full uppercase tracking-wider backdrop-blur-sm shadow-lg">Abandoned</span>
      </div>
    </div>
    
    <div class="flex flex-col flex-1 p-4">
      <h3 class="text-white font-bold text-lg leading-tight line-clamp-2 mb-1" :title="book.title">
        <span v-if="book.series_sequence" class="inline-block px-1.5 py-0.5 bg-brand-primary/20 text-brand-primary text-[10px] font-bold rounded mr-1 align-middle backdrop-blur-sm">
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
