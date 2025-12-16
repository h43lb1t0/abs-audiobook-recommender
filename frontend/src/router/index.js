import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Login from '../views/Login.vue'
// Lazy load others if desired, but direct is fine for small app
import History from '../views/History.vue'
import InProgress from '../views/InProgress.vue'
import ChangePassword from '../views/ChangePassword.vue'

const routes = [
    { path: '/', name: 'Home', component: Home },
    { path: '/login', name: 'Login', component: Login },
    { path: '/history', name: 'History', component: History },
    { path: '/in-progress', name: 'InProgress', component: InProgress },
    { path: '/account', name: 'ChangePassword', component: ChangePassword },
    { path: '/recommend-settings', name: 'RecommendSettings', component: () => import('../views/RecommendSettings.vue') },
    { path: '/admin', name: 'Admin', component: () => import('../views/Admin.vue') },
]

const router = createRouter({
    history: createWebHistory(),
    routes,
})

export default router
