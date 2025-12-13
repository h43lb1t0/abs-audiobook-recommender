<template>
  <div class="flex items-center space-x-1" @mouseleave="hoverRating = 0">
    <button v-for="star in 5" :key="star" type="button" 
      @mouseenter="hoverRating = star"
      @click="$emit('update:rating', star)"
      class="focus:outline-none transition-all duration-200 hover:scale-110 active:scale-95 text-2xl"
      :class="star <= (hoverRating || rating) ? 'text-yellow-400 drop-shadow-[0_0_8px_rgba(250,204,21,0.5)]' : 'text-slate-700'">
      ★
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  rating: {
    type: Number,
    default: 0
  }
})

defineEmits(['update:rating'])

const hoverRating = ref(0)
</script>
