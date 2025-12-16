<template>
  <div class="space-y-6">
    <div class="bg-brand-header shadow px-4 py-5 sm:rounded-lg sm:p-6 border border-white/5">
      <div class="md:grid md:grid-cols-3 md:gap-6">
        <div class="md:col-span-1">
          <h3 class="text-lg font-medium leading-6 text-white">{{ $t('settings.recommendSettings') }}</h3>
          <p class="mt-1 text-sm text-gray-400">
            <!-- Placeholder description -->
            Configuration for recommendation engine.
          </p>
        </div>
        <div class="mt-5 md:mt-0 md:col-span-2">
            <div class="space-y-4">
                 <div>
                    <h3 class="text-base font-medium text-white mb-2">{{ $t('settings.excludeSeriesTitle') }}</h3>
                </div>
                <!-- Search Input -->
                <div>
                    <label for="series-search" class="block text-sm font-medium text-gray-300">{{ $t('settings.searchPlaceholder') }}</label>
                    <div class="mt-1 relative rounded-md shadow-sm">
                        <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <span class="text-gray-500 sm:text-sm">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
                                  <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
                                </svg>
                            </span>
                        </div>
                        <input 
                            type="text" 
                            name="series-search" 
                            id="series-search" 
                            v-model="searchQuery"
                            class="focus:ring-brand-primary focus:border-brand-primary block w-full pl-10 sm:text-sm border-white/10 rounded-md bg-white/5 text-white placeholder-gray-500" 
                            :placeholder="$t('settings.searchPlaceholder')"
                        >
                    </div>
                </div>

                <!-- Series List -->
                <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div v-for="series in filteredSeries" :key="series.id" class="relative rounded-lg border border-white/10 bg-white/5 px-6 py-5 shadow-sm flex items-center space-x-3 hover:border-gray-400 focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-brand-primary">
                        <div class="flex-shrink-0">
                            <img class="h-10 w-10 rounded-full object-cover" :src="`/api/cover/${series.first_book_id}`" alt="">
                        </div>
                        <div class="flex-1 min-w-0">
                            <a href="#" class="focus:outline-none">
                                <span class="absolute inset-0" aria-hidden="true"></span>
                                <p class="text-sm font-medium text-white break-words">{{ series.name }}</p>
                            </a>
                        </div>
                         <!-- Exclude Button (Mock) -->
                         <div class="flex-shrink-0 z-10" @click.stop>
                            <button 
                                @click="toggleExclude(series.id)"
                                :class="[
                                    excludedSeries.has(series.id) ? 'bg-red-500/20 text-red-400 border-red-500/50' : 'bg-white/5 text-gray-300 border-white/10 hover:bg-white/10',
                                    'inline-flex items-center px-2.5 py-1.5 border text-xs font-medium rounded shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-primary transition-all duration-200'
                                ]"
                            >
                                <span v-if="excludedSeries.has(series.id)">{{ $t('settings.excluded') }}</span>
                                <span v-else>
                                    {{ user?.id === 'root' ? $t('settings.excludeEveryone') : $t('settings.exclude') }}
                                </span>
                            </button>
                        </div>
                    </div>
                     <div v-if="filteredSeries.length === 0 && searchQuery" class="col-span-1 sm:col-span-2 text-center text-gray-400 py-4">
                        No series found matching "{{ searchQuery }}"
                    </div>
                </div>
            </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, inject } from 'vue'
import axios from 'axios'

const absUrl = inject('absUrl') // Although we use relative paths for api calls
const user = inject('user')
const seriesList = ref([])
const searchQuery = ref('')
const excludedSeries = ref(new Set()) // Mock Set for excluded IDs

const fetchSeries = async () => {
    try {
        const { data } = await axios.get('/api/series')
        seriesList.value = data
    } catch (e) {
        console.error("Failed to fetch series", e)
    }
}

const filteredSeries = computed(() => {
    if (!searchQuery.value) return []
    const lowerQuery = searchQuery.value.toLowerCase()
    return seriesList.value.filter(s => s.name.toLowerCase().includes(lowerQuery))
})

const toggleExclude = (seriesId) => {
    if (excludedSeries.value.has(seriesId)) {
        excludedSeries.value.delete(seriesId)
    } else {
        excludedSeries.value.add(seriesId)
    }
}

onMounted(() => {
    fetchSeries()
})
</script>
