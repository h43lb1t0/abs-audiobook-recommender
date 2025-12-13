<template>
  <div class="space-y-6 animate-fade-in">
    <div class="flex items-center justify-between">
      <h2 class="text-2xl font-bold text-white">{{ $t('admin.title') }}</h2>
      <button @click="fetchUsers" class="text-brand-primary hover:text-brand-primary/80 transition-colors">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
          <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
        </svg>
      </button>
    </div>

    <div v-if="loading" class="flex justify-center py-12">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-brand-primary"></div>
    </div>

    <div v-else class="bg-brand-card rounded-lg border border-white/5 overflow-hidden">
      <table class="min-w-full divide-y divide-white/10">
        <thead class="bg-black/20">
          <tr>
            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">{{ $t('admin.username') }}</th>
            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">{{ $t('admin.userId') }}</th>
            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">{{ $t('admin.status') }}</th>
            <th scope="col" class="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">{{ $t('admin.actions') }}</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-white/10">
          <tr v-for="user in users" :key="user.id" class="hover:bg-white/5 transition-colors">
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">{{ user.username }}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-400">{{ user.id }}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
               <span v-if="user.id === 'root'" class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-purple-100 text-purple-800">
                Root
              </span>
              <span v-if="user.force_password_change" class="ml-2 px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800">
                {{ $t('admin.resetPending') }}
              </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
              <button @click="openResetModal(user)" class="text-brand-primary hover:text-brand-primary/80 transition-colors">
                {{ $t('admin.resetPassword') }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Reset Password Modal -->
    <div v-if="showModal" class="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
      <div class="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div class="fixed inset-0 bg-gray-900 bg-opacity-75 transition-opacity" aria-hidden="true" @click="closeModal"></div>

        <span class="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

        <div class="inline-block align-bottom bg-brand-header rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full border border-white/10">
          <div class="bg-brand-header px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <div class="sm:flex sm:items-start">
              <div class="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                <h3 class="text-lg leading-6 font-medium text-white" id="modal-title">
                  {{ $t('admin.resetPasswordFor', { username: selectedUser?.username }) }}
                </h3>
                <div class="mt-2">
                  <p class="text-sm text-gray-400">
                    {{ $t('admin.resetPasswordDescription') }}
                  </p>
                </div>
                
                <div class="mt-4 space-y-4">
                    <div v-if="message" :class="`p-2 rounded text-sm ${isError ? 'bg-red-500/10 text-red-400' : 'bg-green-500/10 text-green-400'}`">
                        {{ message }}
                    </div>
                
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-1">{{ $t('admin.newPassword') }}</label>
                        <input type="password" v-model="newPassword" class="block w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-brand-primary focus:border-transparent transition-all">
                    </div>
                </div>
              </div>
            </div>
          </div>
          <div class="bg-black/20 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
            <button type="button" :disabled="processing" @click="resetPassword" class="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-brand-primary text-base font-medium text-white hover:bg-brand-primary/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-primary sm:ml-3 sm:w-auto sm:text-sm disabled:opacity-50 transition-all">
              {{ processing ? $t('admin.processing') : $t('admin.save') }}
            </button>
            <button type="button" @click="closeModal" class="mt-3 w-full inline-flex justify-center rounded-md border border-gray-600 shadow-sm px-4 py-2 bg-transparent text-base font-medium text-gray-300 hover:text-white hover:bg-white/5 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-primary sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm transition-all">
              {{ $t('admin.cancel') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
const users = ref([])
const loading = ref(true)
const showModal = ref(false)
const selectedUser = ref(null)
const newPassword = ref('')
const message = ref('')
const isError = ref(false)
const processing = ref(false)

const fetchUsers = async () => {
    loading.value = true
    try {
        const { data } = await axios.get('/api/admin/users')
        users.value = data
    } catch (e) {
        console.error("Failed to fetch users", e)
    } finally {
        loading.value = false
    }
}

const openResetModal = (user) => {
    selectedUser.value = user
    newPassword.value = ''
    message.value = ''
    isError.value = false
    showModal.value = true
}

const closeModal = () => {
    showModal.value = false
    selectedUser.value = null
}

const resetPassword = async () => {
    if (!newPassword.value) {
        message.value = t('admin.passwordRequired')
        isError.value = true
        return
    }

    if (newPassword.value.length < 4) {
        message.value = t('changePassword.tooShort')
        isError.value = true
        return
    }
    
    processing.value = true
    message.value = ''
    isError.value = false
    
    try {
        await axios.post('/api/admin/change-user-password', {
            user_id: selectedUser.value.id,
            new_password: newPassword.value
        })
        
        message.value = t('admin.success')
        isError.value = false
        
        // Refresh users list to show "Reset Pending" if applicable
        await fetchUsers()
        
        setTimeout(() => {
            closeModal()
        }, 1500)
    } catch (e) {
        console.error(e)
        message.value = e.response?.data?.error || t('admin.error')
        isError.value = true
    } finally {
        processing.value = false
    }
}

onMounted(() => {
    fetchUsers()
})
</script>
