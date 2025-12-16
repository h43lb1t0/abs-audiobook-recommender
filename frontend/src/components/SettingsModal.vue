<template>
  <div class="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
    <div class="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
      <div class="fixed inset-0 bg-gray-900 bg-opacity-75 transition-opacity" aria-hidden="true" @click="$emit('close')"></div>

      <span class="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

      <div class="inline-block align-bottom bg-brand-header rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full border border-white/10">
        <div class="bg-brand-header px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
          <div class="sm:flex sm:items-start">
            <div class="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
              <h3 class="text-lg leading-6 font-medium text-white" id="modal-title">
                {{ $t('settings.title') }}
              </h3>
              <div class="mt-6 border-t border-white/10 pt-4">
                <div class="mb-6">
                    <label class="block text-sm font-medium text-gray-300 mb-2">{{ $t('settings.language') }}</label>
                    <div class="flex space-x-4">
                        <button 
                            @click="changeLocale('en')"
                            :class="[
                                locale === 'en' ? 'bg-brand-primary text-white ring-2 ring-offset-2 ring-offset-brand-header ring-brand-primary' : 'bg-white/5 text-gray-300 hover:bg-white/10',
                                'px-4 py-2 rounded-md text-sm font-medium transition-all duration-200'
                            ]"
                        >
                            English
                        </button>
                        <button 
                            @click="changeLocale('de')"
                            :class="[
                                locale === 'de' ? 'bg-brand-primary text-white ring-2 ring-offset-2 ring-offset-brand-header ring-brand-primary' : 'bg-white/5 text-gray-300 hover:bg-white/10',
                                'px-4 py-2 rounded-md text-sm font-medium transition-all duration-200'
                            ]"
                        >
                            Deutsch
                        </button>
                    </div>
                </div>

                <div class="mb-4">
                    <button @click="goToChangePassword" class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-white/10 hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-primary transition-colors">
                        {{ $t('settings.changePassword') }}
                    </button>
                </div>

                <div class="mb-4">
                    <button @click="goToRecommendSettings" class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-white/10 hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-primary transition-colors">
                        {{ $t('settings.recommendSettings') }}
                    </button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="bg-black/20 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
          <button type="button" class="mt-3 w-full inline-flex justify-center rounded-md border border-gray-600 shadow-sm px-4 py-2 bg-transparent text-base font-medium text-gray-300 hover:text-white hover:bg-white/5 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-primary sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm" @click="$emit('close')">
            {{ $t('settings.close') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import axios from 'axios'

const { locale } = useI18n()
const router = useRouter()
const emit = defineEmits(['close'])

const changeLocale = async (newLocale) => {
    try {
        await axios.post('/api/user/language', { language: newLocale })
    } catch (e) {
        console.error("Failed to save language preference", e)
    }
    // Optimistic update
    locale.value = newLocale
}

const goToChangePassword = () => {
    emit('close')
    router.push('/account')
}

const goToRecommendSettings = () => {
    emit('close')
    router.push('/recommend-settings')
}
</script>
